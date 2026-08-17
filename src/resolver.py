"""
resolver.py — the five-variable site recommender.

Runs LAST, against the fully-built + enriched database and the AutoGraph context graph.

Composes, in one workflow:
  1. VECTOR search  (APPROX_NEAR_COSINE)         -> studies similar to the new protocol   [protocol]
  2. GRAPH traversal (OUTBOUND ... site_topology) -> the facilities that ran them
  3. KEY-VALUE lookup (facility fields)           -> performance, prevalence, age
  4. COMPOSITE scoring across those variables (with a per-variable breakdown)
  5. AutoGraph RETRIEVER call                      -> dosage signal for the matched studies  [dosage]

Four variables (protocol, performance, prevalence, age) come from ONE AQL query.
Dosage comes from the AutoGraph context graph via a separate Retriever call: the DB and
the context graph are two query worlds, and the resolver stitches them.

Missing-data handling: sites missing prevalence/age (non-US or unenriched US) are NOT
excluded. They score on neutral stand-ins and lower the confidence signal.
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import re
import json
import sys
import time
import requests

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
REASONING_MODEL = "gpt-4.1-mini"

# --- Scoring weights (provisional working set; adjust freely) ---
W_PROTOCOL    = 0.35   # semantic similarity of past study to the new protocol
W_PERFORMANCE = 0.30   # site success_score
W_PREVALENCE  = 0.20   # local cancer prevalence (catchment)
W_AGE         = 0.15   # local age fit vs ~40

# Neutral stand-ins when a site lacks geo data
NEUTRAL_PREVALENCE_SCORE = 0.3
NEUTRAL_AGE_SCORE = 0.5

# Depth 1: edges run study -> facility, so the facilities that ran each study are
# exactly one hop out. A second hop reaches investigators, which the facility filter
# discards, so depth 1 returns the same sites without the wasted traversal.
TRAVERSAL_DEPTH = 1

# --- AutoGraph Retriever (dosage) ---
# Read from the environment with no defaults, so a missing value fails loudly instead
# of silently falling back to someone else's deployment. Set these in your .env file
# (see .env.example). rstrip("/") guards against a trailing slash in the host, which
# would otherwise produce a malformed "host/:8529" URL.
RETRIEVER_HOST = os.environ["RETRIEVER_HOST"].rstrip("/")
RETRIEVER_SERVICE_ID = os.environ["RETRIEVER_SERVICE_ID"]
RETRIEVER_POSTFIX = RETRIEVER_SERVICE_ID.split("-")[-1]
RETRIEVER_URL = os.environ.get(
    "RETRIEVER_URL",
    f"{RETRIEVER_HOST}:8529/graphrag/retriever/{RETRIEVER_POSTFIX}/v1/graphrag-query",
)
ARANGO_AUTH_URL = os.environ.get("ARANGO_AUTH_URL", f"{RETRIEVER_HOST}:8529/_open/auth")


# --- Marquee AQL: FOUR variables, returning per-variable sub-scores for the breakdown ---
MARQUEE = """
LET similar = (
    FOR s IN studies
        LET sim = APPROX_NEAR_COSINE(s.embedding, @vec)
        SORT sim DESC
        LIMIT 5
        RETURN {
            nct_id: s.nct_id, title: s.brief_title, phase: s.phase,
            summary: s.summary, eligibility_criteria: s.eligibility_criteria,
            similarity: sim
        }
)

LET sites = (
    FOR study IN similar
        FOR v IN 1..@depth OUTBOUND CONCAT("studies/", study.nct_id) GRAPH "site_topology"
            FILTER IS_SAME_COLLECTION("facilities", v)
            FILTER v.success_score != null
            LET has_geo = v.cancer_prevalence != null
            LET prevalence_score = has_geo ? (v.cancer_prevalence / 15) : @neutral_prev
            LET age_score = (has_geo AND v.median_age != null)
                            ? ((50 - ABS(v.median_age - 40)) / 50) : @neutral_age
            LET composite = (study.similarity * @w_protocol)
                          + (v.success_score  * @w_perf)
                          + (prevalence_score * @w_prev)
                          + (age_score        * @w_age)
            COLLECT
                name = v.name, city = v.city, state = v.state, country = v.country,
                success_score = v.success_score, prevalence = v.cancer_prevalence,
                median_age = v.median_age, has_geo_data = has_geo
            AGGREGATE score = MAX(composite),
                      via_similarity = MAX(study.similarity),
                      prev_sub = MAX(prevalence_score),
                      age_sub  = MAX(age_score)
            SORT score DESC
            LIMIT 10
            RETURN {
                name, city, state, country, success_score,
                prevalence, median_age, has_geo_data,
                composite_score: score, matched_similarity: via_similarity,
                prevalence_score: prev_sub, age_score: age_sub
            }
)

RETURN { similar_studies: similar, recommended_sites: sites }
"""


# Reasoning prompt: explain WHY the top site beat the runner-up, not restate numbers.
REASONING_PROMPT = """
You are a clinical trial feasibility expert. You are given a new protocol and a ranked
list of candidate sites, each with a per-variable score breakdown (protocol similarity,
site success, disease prevalence, age fit). Reason ONLY about the sites provided; never
mention a site that is not in the list, and never invent numbers.

Write 3-4 sentences that explain the TRADEOFFS behind the ranking, not the raw numbers
the reader can already see. In particular, explain why the top site outranked the
runner-up: name the variable where the runner-up was stronger and the variable where the
top site more than made up for it. Note any site missing geographic data, since that
lowers confidence. If dosage evidence is provided, add one sentence saying only whether the
matched trials appear to use standard dosing for this drug class. Do NOT state specific
milligram doses in your prose, since the evidence spans several drugs and a dose could be
misattributed; the dosage evidence is shown separately with exact values. Be specific and
concise. Do not use bullet points. Always put a space after the word "Regarding".
"""


def get_jwt():
    try:
        r = requests.post(
            ARANGO_AUTH_URL,
            json={"username": os.environ["ARANGO_USER"],
                  "password": os.environ["ARANGO_PASSWORD"]},
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("jwt")
    except Exception as e:
        print(f"  [dosage] JWT auth failed: {e}")
        return None


def get_dosage(similar_studies):
    """Query the AutoGraph Retriever for dosage. Returns (summary_lines, raw, doc_count)
    or (None, None, 0). Never raises."""
    token = get_jwt()
    if not token:
        return None, None, 0

    titles = ", ".join(s["title"] for s in similar_studies[:3])
    question = f"What drug dosages are used in trials similar to: {titles}?"
    try:
        r = requests.post(
            RETRIEVER_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"query": question, "query_type": 3, "level": 1, "provider": 0},
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        raw = data.get("result") or data.get("answer") or data.get("response") or ""
        # count cited source documents if the API returns them; else infer from the text
        docs = data.get("sources") or data.get("documents") or []
        doc_count = len(docs) if isinstance(docs, list) else 0
        lines = summarize_dosage(raw)
        return lines, raw, doc_count
    except Exception as e:
        print(f"  [dosage] retriever call failed: {e}")
        return None, None, 0


# Words that signal a captured phrase is a study/description, not a drug name.
_NON_DRUG_TERMS = {
    "summary", "note", "overview", "study", "trial", "phase", "arm", "a phase",
    "background", "conclusion", "results", "methods", "objective", "regarding",
    "these", "the", "this", "dosing", "dosage", "treatment", "patients",
}


def _looks_like_drug(name):
    """A drug name is a short-ish token (usually one word) that isn't a description
    fragment. Reject multi-word phrases and known non-drug lead-ins."""
    low = name.lower().strip()
    if low in _NON_DRUG_TERMS:
        return False
    if any(low.startswith(term + " ") for term in _NON_DRUG_TERMS):
        return False
    # Drug names here are single tokens (ribociclib, abemaciclib, capivasertib).
    # A capture with 3+ words is almost always a description fragment.
    if len(name.split()) > 2:
        return False
    return True


def _ends_mid_word(dose):
    """True if the dose text was cut off mid-sentence: it ends on a bare word with
    no closing punctuation, i.e. not on '.', ')', ']', a digit, or a unit token."""
    tail = dose.rstrip()
    if re.search(r"[.)\]%]$", tail):          # clean sentence/bracket/percent end
        return False
    last_tok = tail.split()[-1] if tail.split() else ""
    # A unit or number as the final token is a legitimate ending (e.g. "150 mg").
    if re.search(r"\d", last_tok) or last_tok.lower() in (
        "mg", "kg", "ml", "bid", "qd", "daily", "weekly", "cycle", "cycles",
        "off", "on", "phases", "arms", "therapy", "weeks", "days", "year", "years",
    ):
        return False
    return True


def _trim_to_last_clause(dose):
    """If the raw retriever answer was cut off mid-word (e.g. '...per Summar',
    '...genetically defined b'), trim back to the last complete clause so a line
    never ends on a fragment. Complete-looking lines are left untouched."""
    if not _ends_mid_word(dose):
        return dose
    # Clause boundaries: sentence period, semicolon, comma, or closing paren + space.
    boundaries = [m.end() for m in re.finditer(r"[.;,)]\s", dose)]
    if not boundaries:
        return dose
    trimmed = dose[:boundaries[-1]].strip().rstrip(",;").strip()
    return trimmed or dose


def _normalize_dose(dose):
    """Strip markdown bold markers, citation refs, and dangling truncated fragments."""
    dose = dose.replace("**", "").strip()
    dose = re.sub(r"\s*\[\d+\]\s*$", "", dose)   # drop trailing [1] style refs
    dose = dose.strip().rstrip(".").strip()
    dose = _trim_to_last_clause(dose)            # cut mid-word truncations cleanly
    return dose


def _dose_key(drug, dose):
    """Dedupe key: same drug + same core mg amount counts once, so 'Ribociclib 600 mg
    once daily' and 'Ribociclib: 600 mg 21 days on' don't double-count."""
    mg = re.search(r"(\d+)\s*mg", dose.lower())
    amount = mg.group(1) if mg else dose.lower()[:12]
    return f"{drug.lower()}|{amount}"


def summarize_dosage(raw):
    """Pull 'Drug: dose' style lines out of the Retriever's markdown answer.
    Keeps retrieval (the raw answer) separate from a clean generated summary.
    Filters description fragments, strips markdown, and dedupes on drug+amount."""
    if not raw:
        return []
    out, seen = [], set()
    for m in re.finditer(
        r"(?:^|\n)[-*\s]*\**([A-Z][A-Za-z0-9\- ]{2,40})\**\s*[:\-]\s*([^\n]{5,160})", raw
    ):
        drug = m.group(1).replace("**", "").strip()
        dose = _normalize_dose(m.group(2))
        # must look like a real dosage (a number + a unit) and a real drug name
        if not re.search(r"\d", dose):
            continue
        if not re.search(r"\bmg\b|/m²|/kg|units?", dose.lower()):
            continue
        if not _looks_like_drug(drug):
            continue
        key = _dose_key(drug, dose)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{drug}: {dose}")
        if len(out) >= 6:
            break
    return out


def assess_confidence(similar_studies, sites, dosage_ok):
    notes = []
    top_sim = max((s["similarity"] for s in similar_studies), default=0)
    if top_sim < 0.6:
        notes.append(f"closest study only {round(top_sim,2)} similar")
    if len(similar_studies) < 3:
        notes.append(f"only {len(similar_studies)} similar studies")
    if len(sites) < 5:
        notes.append(f"only {len(sites)} candidate sites")
    missing_geo = sum(1 for s in sites[:5] if not s.get("has_geo_data"))
    if missing_geo:
        notes.append(f"{missing_geo} of top sites missing geo data")
    if not dosage_ok:
        notes.append("dosage context unavailable")

    if not notes:
        return "high", "strong match, sufficient sites, complete data across all variables"
    if len(notes) == 1:
        return "medium", notes[0]
    return "low", "; ".join(notes)


def score_breakdown(site):
    """Per-variable contribution for one site (values are the raw sub-scores, with weights)."""
    return {
        "protocol_similarity": (round(site["matched_similarity"], 3), W_PROTOCOL),
        "site_success":        (round(site["success_score"], 3),      W_PERFORMANCE),
        "disease_prevalence":  (round(site["prevalence_score"], 3),   W_PREVALENCE),
        "age_match":           (round(site["age_score"], 3),          W_AGE),
    }


def resolve(db, oai, protocol, top_n=3):
    t0 = time.time()
    protocol_text = f"{protocol['title']} {protocol['description']} {protocol['eligibility']}"

    print("embedding protocol...")
    vec = oai.embeddings.create(model=EMBED_MODEL, input=[protocol_text]).data[0].embedding

    # Utility for reproducing the marquee query by hand in the Arango Query Editor:
    # set DUMP_VEC=1 to write the protocol embedding to vec.json, then paste it as the
    # @vec bind variable in the editor. Off by default; does not affect normal runs.
    if os.environ.get("DUMP_VEC") == "1":
        with open("vec.json", "w") as vf:
            json.dump(vec, vf)
        print(f"  [DUMP_VEC] wrote {len(vec)}-dim embedding to vec.json "
              f"(paste as @vec in the Query Editor)")

    print("querying arango (protocol + performance + prevalence + age)...")
    cursor = db.aql.execute(MARQUEE, bind_vars={
        "vec": vec, "depth": TRAVERSAL_DEPTH,
        "w_protocol": W_PROTOCOL, "w_perf": W_PERFORMANCE,
        "w_prev": W_PREVALENCE, "w_age": W_AGE,
        "neutral_prev": NEUTRAL_PREVALENCE_SCORE, "neutral_age": NEUTRAL_AGE_SCORE,
    })
    result = list(cursor)[0]
    similar_studies = result["similar_studies"]
    sites = result["recommended_sites"]
    print(f"found {len(similar_studies)} similar studies, {len(sites)} ranked sites")

    top_sites = sites[:top_n]

    print("retrieving dosage context (AutoGraph)...")
    dosage_lines, dosage_raw, dosage_docs = get_dosage(similar_studies)
    print(f"  dosage: {'ok' if dosage_lines else 'unavailable'}")

    evidence_passages = [
        {"study": s["title"], "similarity": round(s["similarity"], 3),
         "passage": (s.get("summary") or s.get("eligibility_criteria") or "")[:300]}
        for s in similar_studies[:3]
    ]

    print("generating reasoning...")
    evidence = {
        "protocol": protocol_text[:500],
        "ranked_sites": [
            {"name": s["name"], "breakdown": score_breakdown(s),
             "composite": round(s["composite_score"], 3),
             "has_geo_data": s["has_geo_data"]}
            for s in top_sites
        ],
        "dosage_evidence": dosage_lines or "unavailable",
    }
    reasoning = oai.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "system", "content": REASONING_PROMPT},
                  {"role": "user", "content": json.dumps(evidence)}],
    ).choices[0].message.content

    confidence, confidence_notes = assess_confidence(similar_studies, sites, bool(dosage_lines))
    elapsed = round(time.time() - t0, 1)

    stats = {
        "protocol_top_similarity": round(max((s["similarity"] for s in similar_studies), default=0), 3),
        "vector_hits": len(similar_studies),
        "traversal_depth": TRAVERSAL_DEPTH,
        "sites_ranked": len(sites),
        "execution_seconds": elapsed,
    }

    return {
        "recommended_sites": sites, "top_sites": top_sites,
        "evidence_passages": evidence_passages, "reasoning": reasoning,
        "dosage_lines": dosage_lines, "dosage_raw": dosage_raw,
        "confidence": confidence, "confidence_notes": confidence_notes,
        "similar_studies": similar_studies, "stats": stats,
    }


def _bar(weight):
    return "#" * int(round(weight * 10)) + "-" * (10 - int(round(weight * 10)))


if __name__ == "__main__":
    from arango import ArangoClient

    if len(sys.argv) < 2:
        print("usage: python src/resolver.py data/protocol_sample.json")
        sys.exit(1)

    client = ArangoClient(hosts=os.environ["ARANGO_HOST"])
    db = client.db(os.environ["ARANGO_DB"],
                   username=os.environ["ARANGO_USER"],
                   password=os.environ["ARANGO_PASSWORD"])
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(sys.argv[1]) as f:
        protocol = json.load(f)

    result = resolve(db, oai, protocol)

    print("\n=== RECOMMENDED SITES (protocol + performance + prevalence + age) ===")
    for i, s in enumerate(result["top_sites"], 1):
        flag = "" if s.get("has_geo_data") else "  [missing geo data -> lower confidence]"
        print(f"\n{i}. {s['name']} -- {s['city']}, {s['country']}   overall: {round(s['composite_score'],3)}{flag}")
        bd = score_breakdown(s)
        labels = {
            "protocol_similarity": "Protocol similarity",
            "site_success":        "Site success      ",
            "disease_prevalence":  "Disease prevalence",
            "age_match":           "Age match         ",
        }
        for key, (val, wt) in bd.items():
            print(f"   {labels[key]}  {val:<6}  ({int(wt*100)}%)")

    print("\n=== EVIDENCE PASSAGES ===")
    for e in result["evidence_passages"]:
        print(f"- {e['study']} (similarity: {e['similarity']})")

    print("\n=== DOSAGE EVIDENCE (AutoGraph context graph) ===")
    if result["dosage_lines"]:
        for line in result["dosage_lines"]:
            print(f"  - {line}")
        n = len(result["dosage_lines"])
        word = "statement" if n == 1 else "statements"
        print(f"  ({n} distinct dosage {word} extracted from the context graph)")
    else:
        print("  unavailable")

    print("\n=== REASONING ===")
    print(result["reasoning"])

    print("\n=== CONFIDENCE ===")
    print(f"{result['confidence']} -- {result['confidence_notes']}")

    print("\n=== RETRIEVAL STATS ===")
    st = result["stats"]
    print(f"  protocol top similarity : {st['protocol_top_similarity']}")
    print(f"  vector hits             : {st['vector_hits']}")
    print(f"  traversal depth         : {st['traversal_depth']}")
    print(f"  sites ranked            : {st['sites_ranked']}")
    print(f"  execution time          : {st['execution_seconds']} s")

    with open("data/recommendation_output.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nfull output saved to data/recommendation_output.json")