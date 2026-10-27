"""
resolver.py — the site recommender over four ranked variables plus dosage.

Runs LAST, against the fully-built + enriched database and the AutoGraph context graph.

Composes, in one workflow:
  1. VECTOR search  (APPROX_NEAR_COSINE)         -> studies similar to the new protocol   [protocol]
  2. GRAPH traversal (OUTBOUND ... site_topology) -> the facilities that ran them
  3. KEY-VALUE lookup (facility fields)           -> activity, prevalence, age
  4. COMPOSITE scoring across those variables (with a per-variable breakdown)
  5. AutoGraph RETRIEVER call                      -> dosage signal for the matched studies  [dosage]

Four variables (protocol, activity, prevalence, age) come from ONE AQL query.
Dosage comes from the AutoGraph context graph via a separate Retriever call: the DB and
the context graph are two query worlds, and the resolver stitches them.

Missing-data handling: sites missing prevalence/age (non-US or unenriched US) are NOT
excluded. They score on neutral stand-ins and lower the confidence signal.

The median-age signal is synthetic (see enrich.synthetic_median_age) and is labelled as such
in the output so a reader never mistakes it for a measured value. The old random success_score
has been removed from the ranking entirely.
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
REASONING_MODEL = "gpt-5-mini"

# --- Scoring weights (must sum to 1.0) ---
# Protocol relevance, local prevalence, and site activity are real signals:
# protocol relevance and local prevalence carry the ordering, and site activity
# is a real count of the active+recruiting interventional trials the hospital
# runs. The age signal (20%) is a synthetic stand-in (see AGE_SYNTHETIC and
# enrich.py), labelled as such in the output; the old random success_score was
# removed from the ranking entirely.
W_PROTOCOL    = 0.45   # per-site aggregate relevance of the matched studies
W_ACTIVITY    = 0.10   # site activity: log-scaled count of active+recruiting trials
W_PREVALENCE  = 0.25   # local cancer prevalence (catchment)
W_AGE         = 0.20   # local age fit vs ~40 (see AGE_SYNTHETIC)

# Reference prevalence (percent) that maps to a prevalence sub-score of 1.0.
# CDC PLACES all-cancer prevalence tops out near this value, so dividing by it
# normalizes the raw percentage into roughly 0..1 for the composite.
PREVALENCE_REF = 15

# Reference trial count that maps to a site-activity sub-score of 1.0. Activity is
# log-scaled (LOG(1+count)/LOG(1+ref)) so a very busy site does not dominate.
# Must be >= 1: at 0 the AQL denominator LOG(1+ref) is 0 and the score divides by
# zero.
ACTIVITY_REF = 50
assert ACTIVITY_REF >= 1, "ACTIVITY_REF must be >= 1 (LOG(1+ref) is the activity denominator)"

# The median-age signal is still a synthetic stand-in (see enrich.py). It is
# labelled as such in the output; set to False once a real county-age source is
# wired in.
AGE_SYNTHETIC = True

# Minimum top-similarity for a protocol to be considered on-topic. Below this the
# closest past study is only loosely related, so the finding is low confidence.
# Breast protocols score ~0.77-0.83 here; an off-topic (e.g. lung) protocol fell
# at ~0.67, so the bar sits between them.
MIN_TOP_SIMILARITY = 0.75

# Placeholder facility names (e.g. "Research Site", "Local Institution - 0042")
# are excluded from ranking. The exclusion is done inline in the marquee AQL
# (see the FILTER LOWER(v.name) ... block) so it runs in the database next to the
# traversal; it is kept in one place there rather than duplicated as a Python
# regex that could drift out of sync.

# Neutral stand-ins when a site lacks geo data
NEUTRAL_PREVALENCE_SCORE = 0.3
NEUTRAL_AGE_SCORE = 0.5

# When the top two composite scores are within this margin the ordering between
# them is within noise, so the finding is reported as at most "medium" and the
# tie is stated plainly rather than presented as a confident winner.
NEAR_TIE_EPS = 0.02

# Depth 1: edges run study -> facility, so the facilities that ran each study are
# exactly one hop out. A second hop reaches investigators, which the facility filter
# discards, so depth 1 returns the same sites without the wasted traversal.
TRAVERSAL_DEPTH = 1

# The marquee returns this many ranked facility rows; the resolver dedups
# near-duplicate records down to distinct sites before taking the top N, so the
# buffer must exceed top_n with room for duplicates.
SITE_BUFFER = 50

# --- AutoGraph Retriever (dosage) ---
# Read from the environment with no defaults, so a missing value fails loudly instead
# of silently falling back to someone else's deployment. Set these in your .env file
# (see .env.example).
def _host_with_port(base, port=8529):
    """Return the host carrying the Arango API port. The port is added only when
    the host has no explicit port; a host that already names a port, or that
    carries a path, is left untouched, so a non-default deployment URL is never
    mangled into 'host:port/path:8529'. Set RETRIEVER_URL directly to override."""
    b = (base or "").rstrip("/")
    m = re.match(r"^(https?://[^/:]+)(:\d+)?$", b)
    return f"{m.group(1)}:{port}" if (m and not m.group(2)) else b


RETRIEVER_HOST = os.environ["RETRIEVER_HOST"].rstrip("/")
RETRIEVER_SERVICE_ID = os.environ["RETRIEVER_SERVICE_ID"]
RETRIEVER_POSTFIX = RETRIEVER_SERVICE_ID.split("-")[-1]
RETRIEVER_URL = os.environ.get(
    "RETRIEVER_URL",
    f"{_host_with_port(RETRIEVER_HOST)}/graphrag/retriever/{RETRIEVER_POSTFIX}/v1/graphrag-query",
)
ARANGO_AUTH_URL = os.environ.get("ARANGO_AUTH_URL", f"{_host_with_port(RETRIEVER_HOST)}/_open/auth")


# --- Marquee AQL: FOUR variables, returning per-variable sub-scores for the breakdown ---
MARQUEE = """
LET similar = (
    FOR s IN studies
        // nProbe must be >= the index's nLists (ingest.VECTOR_NLISTS, currently 2)
        // so the search probes every inverted list; the default of 1 scans only one
        // list and can miss the true nearest neighbours (e.g. the top PALLAS match).
        LET sim = APPROX_NEAR_COSINE(s.embedding, @vec, { nProbe: 2 })
        SORT sim DESC
        LIMIT 5
        RETURN {
            nct_id: s.nct_id, title: s.brief_title, phase: s.phase,
            summary: s.summary, eligibility_criteria: s.eligibility_criteria,
            similarity: sim
        }
)

// Stage 1: one row per (hospital, trial). AACT can list the same hospital
// more than once inside a single trial (several facility records / edges),
// which would otherwise feed that trial's similarity into the noisy-or below
// several times and inflate a site's protocol score. Collapsing to a distinct
// (hospital identity, nct_id) pair first makes the noisy-or count each trial
// exactly once.
LET site_trials = (
    FOR study IN similar
        FOR v IN 1..@depth OUTBOUND CONCAT("studies/", study.nct_id) GRAPH "site_topology"
            FILTER IS_SAME_COLLECTION("facilities", v)
            FILTER v.name != null
            // Drop placeholder facility rows so they neither rank nor collapse
            // into one fake hospital.
            FILTER LOWER(v.name) NOT IN ["research site", "investigative site",
                                         "clinical research site", "site reference id"]
            FILTER LOWER(v.name) NOT LIKE "local institution%"
            FILTER LOWER(v.name) NOT LIKE "site %"
            COLLECT
                name = v.name, city = v.city, state = v.state, country = v.country,
                prevalence = v.cancer_prevalence,
                median_age = v.median_age, has_geo_data = (v.cancer_prevalence != null),
                nct_id = study.nct_id
            AGGREGATE sim = MAX(study.similarity)
            RETURN {
                name, city, state, country, prevalence, median_age, has_geo_data,
                nct_id, sim
            }
)

LET sites = (
    FOR r IN site_trials
            LET has_geo = r.has_geo_data
            // Clamp both geo sub-scores to [0,1] so they stay on the same scale as
            // the cosine similarity. Without this a site above @prev_ref prevalence
            // would score >1 and a median age far from the anchor would score <0,
            // which would let one variable outvote the weighted composite.
            LET prevalence_score = has_geo
                            ? MIN([1, MAX([0, r.prevalence / @prev_ref])]) : @neutral_prev
            LET age_score = (has_geo AND r.median_age != null)
                            ? MIN([1, MAX([0, (50 - ABS(r.median_age - 40)) / 50])]) : @neutral_age
            // Group on the facility's identity, NOT on any per-row value, so every
            // matched trial for the same facility lands in one group and the
            // noisy-or below can combine them. Each row here is already one
            // distinct trial per hospital (stage 1), so no trial is double counted.
            COLLECT
                name = r.name, city = r.city, state = r.state, country = r.country,
                prevalence = r.prevalence,
                median_age = r.median_age, has_geo_data = has_geo
            AGGREGATE log_miss      = SUM(LOG(1 - MIN([0.999, MAX([0, r.sim])]))),
                      best_sim      = MAX(r.sim),
                      matched_count = COUNT_DISTINCT(r.nct_id),
                      prev_sub      = MAX(prevalence_score),
                      age_sub       = MAX(age_score)
            // Site activity: distinct active+recruiting interventional trials across
            // ALL facility records for this physical hospital (AACT stores one
            // facility record per study, so counting per record would always be ~1;
            // counting by hospital identity gives the real per-hospital activity).
            // Match the SAME identity the COLLECT groups on (name+city+state+
            // country) so the activity count is for exactly this grouped site.
            // The studies collection is loaded active+recruiting interventional
            // only (see ingest.py), but the status/type filter is restated here
            // so the "active trials" label stays true even if the DB is widened.
            LET trial_count = LENGTH(
                FOR f IN facilities
                    FILTER f.name == name AND f.city == city
                       AND f.state == state AND f.country == country
                    FOR st IN 1..1 INBOUND f._id GRAPH "site_topology"
                        FILTER IS_SAME_COLLECTION("studies", st)
                        FILTER st.study_type == "INTERVENTIONAL"
                        FILTER st.overall_status IN ["ACTIVE_NOT_RECRUITING", "RECRUITING"]
                        RETURN DISTINCT st._key
            )
            // Protocol is a PER-SITE signal, not a single best match: a site that
            // ran several of the top-k similar studies should outrank one that ran
            // a single trial. Bounded noisy-or over the matched similarities,
            // protocol = 1 - PRODUCT(1 - sim_i) = 1 - EXP(SUM(LOG(1 - sim_i))).
            LET protocol_score = 1 - EXP(log_miss)
            // Site activity, log-scaled so a very busy site does not dominate.
            LET activity_score = MIN([1, MAX([0, LOG(1 + trial_count) / LOG(1 + @activity_ref)])])
            LET composite = (protocol_score * @w_protocol)
                          + (activity_score * @w_activity)
                          + (prev_sub       * @w_prev)
                          + (age_sub        * @w_age)
            SORT composite DESC, matched_count DESC, best_sim DESC
            LIMIT @site_buffer
            RETURN {
                name, city, state, country,
                prevalence, median_age, has_geo_data,
                trial_count: trial_count, activity_score: activity_score,
                composite_score: composite,
                matched_similarity: protocol_score,
                best_similarity: best_sim,
                matched_count: matched_count,
                prevalence_score: prev_sub, age_score: age_sub
            }
)

RETURN { similar_studies: similar, recommended_sites: sites }
"""


# Reasoning prompt: explain the tradeoffs behind the ranking, grounded only in the
# data provided. The model must not invent anything and must say when it does not know.
REASONING_PROMPT = """
You are a clinical trial feasibility expert. You are given a new protocol and a ranked
list of candidate sites, each with a per-variable score breakdown (protocol similarity,
site activity, disease prevalence, age fit) and a composite score.

Honesty rules (follow all of them):
- Use ONLY the sites, numbers, and evidence you are given. Never mention a site that is not
  in the list, and never state a number that is not in the data provided.
- Do not infer, estimate, or guess any value. If something is not in the data, do not
  claim it. When a site is missing geographic data, say that it is missing and that this
  lowers confidence; do not fill the gap with an assumption.
- The ranking is fixed and authoritative: the first site is the top recommendation. Explain
  it; do not re-rank it, and never present a lower-ranked site as the overall winner.
- You are told only whether dosage information is available (dosage_available: true or
  false). You are NOT given the doses. If true, you may say at most one sentence: that
  dosage information was retrieved and is listed separately. If false, say plainly
  that dosage context was unavailable. Never state a dose, a drug name, or a schedule, and
  never claim the dosing is "standard" or "non-standard": you do not have that information.

Then write 3-4 sentences explaining the tradeoffs behind the ranking: name the one variable
where the runner-up scored higher than the top site and the variable where the top site made
up the gap. Be specific and concise. Do not use bullet points.
"""


def _extract_answer(data):
    """Pull the answer text out of whatever shape the Retriever returns.
    Tries the common top-level keys, then one level of nesting, so a change in
    the response envelope does not silently blank the dosage out."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for k in ("result", "answer", "response", "output", "text", "content"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for k in ("data", "result", "response"):
            v = data.get(k)
            if isinstance(v, (dict, list)):
                inner = _extract_answer(v)
                if inner:
                    return inner
        return ""
    if isinstance(data, list):
        # A response may wrap the answer in a list of chunks/messages. Pull text
        # out of each element and join, rather than stringifying the raw list
        # (which would feed Python repr noise into the dose parser).
        parts = [_extract_answer(item) for item in data]
        return "\n".join(p for p in parts if p)
    return ""


def _post_with_retry(url, attempts=3, backoff=1.5, **kwargs):
    """POST with a few retries on transient failures (timeouts, cold starts, 5xx),
    so an intermittent Retriever or auth hiccup does not surface as 'no dosage'."""
    last = None
    for i in range(attempts):
        try:
            r = requests.post(url, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(backoff * (i + 1))
    raise last


def get_jwt():
    try:
        r = _post_with_retry(
            ARANGO_AUTH_URL,
            json={"username": os.environ["ARANGO_USER"],
                  "password": os.environ["ARANGO_PASSWORD"]},
            timeout=20,
        )
        return r.json().get("jwt")
    except Exception as e:
        print(f"  [dosage] JWT auth failed: {e}")
        return None


def get_dosage(similar_studies):
    """Query the AutoGraph Retriever for dosage. Returns (summary_lines, raw)
    or (None, None). Never raises."""
    token = get_jwt()
    if not token:
        return None, None

    # Query the Retriever by the matched trials' NCT IDs (and titles as a hint),
    # so the dosage answer is tied to these specific trials rather than to loosely
    # related text. The corpus is built from the same trial set (see export_docs.py).
    ids = ", ".join(s["nct_id"] for s in similar_studies[:5] if s.get("nct_id"))
    titles = "; ".join(s["title"] for s in similar_studies[:3])
    question = (f"What drug dosages are used in these clinical trials, by NCT ID: {ids}? "
                f"The trials are: {titles}. Report the drug and dose for each.")
    try:
        r = _post_with_retry(
            RETRIEVER_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            # query_type 3 = Unified Search (vector + graph); level 1 = one-hop
            # expansion; provider 0 = the default configured LLM provider.
            json={"query": question, "query_type": 3, "level": 1, "provider": 0},
            timeout=180,
        )
        data = r.json()
        raw = _extract_answer(data)
        # Accept only the configured drug class as plain-word drug names, so a prose
        # answer cannot pass a narration word off as a drug. Empty/unknown -> shape rule.
        try:
            from export_docs import DRUG_KEYWORDS as _DK
            known = {d.lower() for d in _DK} or None
        except Exception:
            known = None
        lines = summarize_dosage(raw, known)
        return lines, raw
    except Exception as e:
        print(f"  [dosage] retriever call failed: {e}")
        return None, None


# Deterministic local fallback directory: the per-trial docs export_docs.py wrote.
CORPUS_DIR_DEFAULT = "data/autograph_docs"


def get_dosage_from_corpus(similar_studies):
    """Deterministic local fallback for dosage, used when the AutoGraph Retriever
    returns nothing parseable (LLM variance or a transient failure). Reads the
    matched trials' own intervention text (the files export_docs.py wrote) and
    pairs each drug-class drug with the dose stated beside it. No network and no
    LLM, so it is reproducible run to run. Returns a list of 'Drug: dose' lines."""
    try:
        from export_docs import DRUG_KEYWORDS as _DRUGS, OUT_DIR as _DIR
    except Exception:
        _DRUGS = []
        _DIR = os.environ.get("AUTOGRAPH_DOCS_DIR", CORPUS_DIR_DEFAULT)
    if not _DRUGS:
        return []
    out, seen = [], set()
    for st in similar_studies[:5]:
        nct = st.get("nct_id")
        if not nct:
            continue
        try:
            text = open(os.path.join(_DIR, f"{nct}.txt"), encoding="utf-8").read()
        except OSError:
            continue
        low = text.lower()
        for drug in _DRUGS:
            pos = low.find(drug.lower())
            while pos != -1:
                window = text[pos:pos + 160]       # original-case, drug-led
                m = _DOSE_SHAPE_RE.search(window)
                if m:
                    snippet = _normalize_dose(window[m.start():m.start() + 120])
                    # Tidy common AACT free-text noise: a leading "125mg:" style
                    # duplicate label and a restated drug name.
                    snippet = re.sub(r"^\d+\s*m?g\s*:\s*", "", snippet, flags=re.IGNORECASE)
                    snippet = re.sub(r"^" + re.escape(drug) + r"\s+(at a dose of\s+)?", "",
                                     snippet, flags=re.IGNORECASE).strip()
                    if _DOSE_SHAPE_RE.search(snippet):
                        key = _dose_key(drug, snippet)
                        if key not in seen:
                            seen.add(key)
                            out.append(f"{drug.capitalize()}: {snippet}")
                            break               # one dose per drug per trial
                pos = low.find(drug.lower(), pos + 1)
            if len(out) >= MAX_DOSE_LINES:
                break
        if len(out) >= MAX_DOSE_LINES:
            break
    return out[:MAX_DOSE_LINES]


# Words that signal a captured phrase is a study/description or a field label,
# not a drug name.
_NON_DRUG_TERMS = {
    "summary", "note", "overview", "study", "trial", "phase", "arm", "a phase",
    "background", "conclusion", "results", "methods", "objective", "regarding",
    "these", "the", "this", "dosing", "dosage", "treatment", "patients",
    "drug", "drugs", "dose", "doses", "medication", "medications", "agent",
    "intervention", "drug name", "regimen", "notes", "context",
    "given", "taken", "administered", "received", "using", "per", "findings",
}


def _ends_mid_word(dose):
    """True if the dose text was cut off mid-sentence: it ends on a bare word with
    no closing punctuation, i.e. not on '.', ')', ']', a digit, or a unit token."""
    tail = dose.rstrip()
    low_tail = tail.lower()
    # An abbreviation like "e.g." or a dangling open paren is NOT a clean end,
    # even though it ends on a period, so treat it as mid-sentence.
    if low_tail.endswith(("e.g.", "i.e.", "(e.g.", "(i.e.")) or tail.endswith("("):
        return True
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
    dose = re.sub(r"\s*\[\d+\]\s*", " ", dose)   # drop [1] style citation refs anywhere
    dose = re.sub(r"^\s*(dose|dosage)\s*:\s*", "", dose, flags=re.IGNORECASE)  # strip a "Dose:" label
    dose = re.sub(r"\s*\([^)]*$", "", dose)      # drop a dangling "(e.g" style open-paren fragment
    dose = re.sub(r"\s{2,}", " ", dose).strip().rstrip(".").strip()
    dose = _trim_to_last_clause(dose)            # cut mid-word truncations cleanly
    return dose


def _dose_key(drug, dose):
    """Dedupe key: same drug + same core mg amount counts once, so 'Ribociclib 600 mg
    once daily' and 'Ribociclib: 600 mg 21 days on' don't double-count."""
    mg = re.search(r"(\d+)\s*mg", dose.lower())
    amount = mg.group(1) if mg else dose.lower()[:12]
    return f"{drug.lower()}|{amount}"


# A dose line must carry a number followed by a recognized unit to be kept.
_DOSE_SHAPE_RE = re.compile(r"\d+(\.\d+)?\s*(mg|mcg|g|ml|/m²|/m2|/kg|units?)\b", re.IGNORECASE)

# An NCT id prefix the Retriever may put before the real drug name
# (e.g. "NCT05069038: Palbociclib ..."); used to skip it when scanning for drugs.
_NCT_PREFIX_RE = re.compile(r"^nct\d{5,}", re.IGNORECASE)
MAX_DOSE_LINES = 6


def _plausible_drug_token(tok, known_drugs=None):
    """A token is a plausible drug name when it is not a stop-word or an NCT id and
    it either carries a drug-code shape (a digit or hyphen, e.g. ARV-471), is a known
    drug from the matched set, or -- when no known set is supplied (the unit tests) --
    is a reasonably long word. Requiring a known drug for plain words stops ordinary
    prose words ("Given", "Taken") that happen to sit before a dose from being read as
    a drug name."""
    low = tok.lower().strip(":")
    if not low or low in _NON_DRUG_TERMS or _NCT_PREFIX_RE.match(low):
        return False
    if not re.match(r"^[a-z][a-z0-9/()‐‑\-]*$", low):
        return False
    if any(c.isdigit() for c in low) or "-" in low:
        return True
    if known_drugs is not None:
        stem = re.split(r"[\s/()]", low)[0]
        return low in known_drugs or stem in known_drugs
    return len(low) >= 5


def summarize_dosage(raw, known_drugs=None):
    """Extract 'Drug: dose' lines from the Retriever's answer, anchored on the DOSE.
    For each quantity found, the drug is the nearest plausible drug token before it
    on the same line, so any narration prefix ("Findings:", "Notes:"), trial id
    ("NCT0...:") or field label is skipped structurally rather than by a word list.
    Keeps the raw answer separate from this clean summary, dedupes on drug + core
    amount, and caps the number of lines."""
    if not raw:
        return []
    out, seen = [], set()
    for line in raw.splitlines():
        # Pre-scan all doses on the line so each dose window can end at the start
        # of the NEXT dose, not run a fixed 140 chars forward. Without this, a
        # line listing several drugs ("Palbociclib 125 mg, letrozole 2.5 mg,
        # anastrozole 1 mg") swallows the following drugs' doses into the first
        # drug's entry (so "2.5 mg" / "1 mg" bleed into the palbociclib line).
        matches = list(_DOSE_SHAPE_RE.finditer(line))
        prev_dose_end = 0
        for idx, dm in enumerate(matches):
            # Only look for the drug name in the text since the previous dose on
            # this line, so a line listing several drugs attributes each dose to
            # the drug stated just before it instead of falling back to the first.
            prefix = line[prev_dose_end:dm.start()]
            prev_dose_end = dm.end()
            toks = re.findall(r"[A-Za-z][A-Za-z0-9/()‐‑\-]*", prefix)
            drug = next((t for t in reversed(toks) if _plausible_drug_token(t, known_drugs)), None)
            if not drug:
                continue
            # Cap this dose at the next dose on the line (or 140 chars forward,
            # whichever is nearer), so a following drug's amount never leaks in.
            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            dose_end = min(dm.start() + 140, next_start)
            dose = _normalize_dose(line[dm.start():dose_end])
            if not _DOSE_SHAPE_RE.search(dose):
                continue
            key = _dose_key(drug, dose)
            if key in seen:
                continue
            seen.add(key)
            drug = drug if any(c.isupper() for c in drug) else drug.capitalize()
            out.append(f"{drug}: {dose}")
            if len(out) >= MAX_DOSE_LINES:
                return out
    return out


def _dosage_on_topic(dose_lines, similar_studies):
    """Heuristic: does the EXTRACTED dosage actually concern the matched trials'
    drugs? True when at least one parsed dose line names a drug that appears in a
    matched study's title or summary.

    This runs on the parsed `dose_lines` (drug: dose), not on the raw Retriever
    answer. The raw answer echoes the NCT ids from the question, so a check
    against the raw text passes even for a refusal or an off-topic answer; a
    drug-name match against the matched trials' own text does not. Guards against
    a dosage corpus that returns doses for drugs unrelated to this protocol."""
    if not dose_lines:
        return False
    corpus = " ".join(
        f"{s.get('title', '')} {s.get('summary', '')}" for s in similar_studies
    ).lower()
    if not corpus.strip():
        return False
    for line in dose_lines:
        drug = line.split(":", 1)[0].strip().lower()
        if not drug:
            continue
        if drug in corpus:
            return True
        # Fall back to the drug-name stem so a hyphenated or suffixed name
        # (e.g. "ARV-471", "palbociclib isethionate") still matches.
        stem = re.split(r"[\s\-]", drug)[0]
        if len(stem) >= 4 and stem in corpus:
            return True
    return False


def assess_confidence(similar_studies, sites, dosage_ok, dosage_on_topic=True):
    notes = []
    top_sim = max((s["similarity"] for s in similar_studies), default=0)

    # On-topic gate: below MIN_TOP_SIMILARITY the closest past study is only
    # loosely related, so this is not a protocol the corpus can speak to.
    if top_sim < MIN_TOP_SIMILARITY:
        notes.append(f"closest study only {round(top_sim, 2)} similar "
                     f"(below {MIN_TOP_SIMILARITY}); protocol may be off-topic")

    missing_geo = sum(1 for s in sites[:5] if not s.get("has_geo_data"))
    if missing_geo:
        notes.append(f"{missing_geo} of top sites missing geo data")

    if not dosage_ok:
        notes.append("dosage context unavailable")
    elif not dosage_on_topic:
        notes.append("dosage evidence does not reference the matched trials")

    # Margin-aware: if the top two composites are within NEAR_TIE_EPS, the winner
    # is within noise, so never report "high" on that basis; state the tie.
    if len(sites) >= 2 and abs(sites[0]["composite_score"] - sites[1]["composite_score"]) < NEAR_TIE_EPS:
        notes.append(f"top two sites effectively tied (composites within {NEAR_TIE_EPS}); "
                     f"the ordering between them is within noise")

    if not notes:
        return "high", ("strong match, sufficient sites, complete available data "
                        "across all variables (age is a synthetic stand-in)")
    if len(notes) == 1:
        return "medium", notes[0]
    return "low", "; ".join(notes)


def score_breakdown(site):
    """Per-variable contribution for one site (values are the raw sub-scores, with weights)."""
    return {
        "protocol_similarity": (round(site["matched_similarity"], 3), W_PROTOCOL),
        "site_activity":       (round(site["activity_score"], 3),     W_ACTIVITY),
        "disease_prevalence":  (round(site["prevalence_score"], 3),   W_PREVALENCE),
        "age_match":           (round(site["age_score"], 3),          W_AGE),
    }


_TOP_CLAIM_RE = re.compile(
    r"(ranks?\s+first|top\s+(?:choice|site|pick|recommendation|candidate)|"
    r"strongest\s+overall|best\s+(?:choice|overall|site|candidate)|"
    r"number\s+one|#\s*1|most\s+suitable\s+overall)", re.IGNORECASE)

# Generic words in facility names; they do not identify a specific site on their own.
_SITE_STOPWORDS = {
    "cancer", "center", "centre", "institute", "hospital", "medical", "health",
    "clinic", "clinical", "university", "of", "the", "and", "at", "for", "care",
    "oncology", "group", "associates", "regional", "memorial", "system", "services",
}


def _site_tokens(name):
    """Distinctive lowercase tokens of a site name (drops generic words like
    'Cancer' or 'Center'), so a shortened mention like 'Avera' still matches
    'Avera Cancer Institute-Aberdeen'."""
    toks = [re.sub(r"[^a-z0-9]", "", t) for t in (name or "").lower().split()]
    toks = [t for t in toks if len(t) >= 3 and t not in _SITE_STOPWORDS]
    return toks or [re.sub(r"[^a-z0-9]", "", (name or "").lower())]


def _dedup_key(site):
    """Collapse near-duplicate facility records to one physical site: the sorted
    distinctive name tokens plus city and country. 'Marin Cancer Center' and
    'Marin Cancer Care' in the same city map to the same key.

    Note: this deliberately drops state and uses name TOKENS, not the exact
    name+city+state+country identity that the scoring COLLECT and trial_count use.
    The looseness is intentional here: this step merges records that differ only
    by name spelling/suffix for one physical place, which the exact identity would
    keep as separate rows. The two identities serve different purposes, so they
    are not meant to match."""
    stem = "".join(sorted(_site_tokens(site.get("name", ""))))
    city = (site.get("city") or "").strip().lower()
    country = (site.get("country") or "").strip().lower()
    return (stem, city, country)


def dedup_sites(sites):
    """Drop near-duplicate facilities from an already-ranked list, keeping the
    highest-scoring record for each physical site. AACT carries several records
    for one site (see the README's entity-resolution note), which would otherwise
    pad the ranking with the same place twice."""
    seen, out = set(), []
    for s in sites:
        key = _dedup_key(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _reasoning_sound(reasoning, top_sites):
    """Accept the model's reasoning unless it actually contradicts the ranking.

    The ranking is fixed and authoritative; this only rejects prose that fights it.
    Reasoning is unsound when the top site is not mentioned at all, or when a
    lower-ranked site is explicitly called the best / top / overall choice. Site
    names are matched on distinctive tokens, so a natural shortening ('Avera' for
    'Avera Cancer Institute-Aberdeen') still counts as a mention. Merely noting a
    runner-up is stronger on one variable is allowed, so a genuine tradeoff
    explanation is kept. A rejected explanation falls back to the deterministic
    summary, never to a fabricated one."""
    if not top_sites:
        return True
    low = (reasoning or "").lower()

    # The top site must be mentioned (by any of its distinctive tokens).
    top_tokens = set(_site_tokens(top_sites[0]["name"]))
    if not any(t in low for t in top_tokens):
        return False

    # Reject only if a lower-ranked site is explicitly crowned the overall winner.
    # Tokens a lower site shares with the top site (e.g. a near-duplicate facility
    # like "Marin Cancer Center" vs "Marin Cancer Care") are skipped, so a shared
    # name near a ranking phrase about the top site is not read as an inversion.
    for s in top_sites[1:]:
        for tok in _site_tokens(s["name"]):
            if tok in top_tokens:
                continue
            start = low.find(tok)
            while start != -1:
                window = low[max(0, start - 40):start + len(tok) + 40]
                if _TOP_CLAIM_RE.search(window):
                    return False
                start = low.find(tok, start + 1)
    return True


def _fallback_reasoning(top_sites):
    """Deterministic summary used when the generated reasoning does not lead with
    the top-ranked site."""
    if not top_sites:
        return "No candidate sites were returned for this protocol."
    top = top_sites[0]
    line = (f"{top['name']} ranks first with a composite score of "
            f"{round(top['composite_score'], 3)} across protocol relevance, site activity, "
            f"disease prevalence, and age fit.")
    if len(top_sites) > 1:
        runner = top_sites[1]
        line += (f" It is followed by {runner['name']} at "
                 f"{round(runner['composite_score'], 3)}.")
    line += " See the per-variable breakdown above for how each site compares."
    return line


_REQUIRED_PROTOCOL_FIELDS = ("title", "description", "eligibility")


def validate_protocol(protocol):
    """Fail loudly on a malformed protocol BEFORE it reaches the embedding call.
    A missing field would otherwise surface as a cryptic KeyError deep in resolve(),
    and an empty field would embed a half-blank string and return a confident but
    meaningless ranking -- the worst failure mode, because it looks like a result."""
    if not isinstance(protocol, dict):
        raise ValueError(f"protocol must be a JSON object, got {type(protocol).__name__}")
    missing = [f for f in _REQUIRED_PROTOCOL_FIELDS if f not in protocol]
    if missing:
        raise ValueError(f"protocol is missing required field(s): {', '.join(missing)}")
    blank = [f for f in _REQUIRED_PROTOCOL_FIELDS
             if not isinstance(protocol[f], str) or not protocol[f].strip()]
    if blank:
        raise ValueError(f"protocol field(s) must be non-empty text: {', '.join(blank)}")
    # Soft signal only: a very short field usually means a malformed protocol that
    # will embed poorly. Warn, don't fail -- a terse protocol may still be valid.
    short = [f for f in _REQUIRED_PROTOCOL_FIELDS if len(protocol[f].strip()) < 20]
    if short:
        print(f"  warning: protocol field(s) look unusually short: {', '.join(short)}")


def _studies_are_embedded(db):
    """True if at least one study carries an embedding. Guards against running
    against an unbuilt/empty database, which would otherwise return an empty or
    confusing ranking instead of a clear 'build the DB first' error."""
    return bool(list(db.aql.execute(
        "RETURN LENGTH(FOR s IN studies FILTER s.embedding != null LIMIT 1 RETURN 1)"))[0])


def resolve(db, oai, protocol, top_n=3):
    t0 = time.time()
    validate_protocol(protocol)
    if not _studies_are_embedded(db):
        raise RuntimeError(
            "no embedded studies in the database -- build it first with "
            "`python src/ingest.py` (then enrich.py).")
    protocol_text = f"{protocol['title']} {protocol['description']} {protocol['eligibility']}"

    print("embedding protocol...")
    # The embedding is required to query at all. Fail with a clear message rather
    # than a raw traceback on a transient API error (same posture as get_dosage,
    # which never crashes the run).
    try:
        vec = oai.embeddings.create(model=EMBED_MODEL, input=[protocol_text]).data[0].embedding
    except Exception as e:
        raise RuntimeError(f"could not embed the protocol (OpenAI embeddings call failed): {e}") from None

    # Utility for reproducing the marquee query by hand in the Arango Query Editor:
    # set DUMP_VEC=1 to write the protocol embedding to vec.json, then paste it as the
    # @vec bind variable in the editor. Off by default; does not affect normal runs.
    if os.environ.get("DUMP_VEC") == "1":
        with open("vec.json", "w") as vf:
            json.dump(vec, vf)
        print(f"  [DUMP_VEC] wrote {len(vec)}-dim embedding to vec.json "
              f"(paste as @vec in the Query Editor)")

    print("querying arango (protocol + activity + prevalence + age)...")
    cursor = db.aql.execute(MARQUEE, bind_vars={
        "vec": vec, "depth": TRAVERSAL_DEPTH,
        "w_protocol": W_PROTOCOL, "w_activity": W_ACTIVITY,
        "w_prev": W_PREVALENCE, "w_age": W_AGE,
        "prev_ref": PREVALENCE_REF, "activity_ref": ACTIVITY_REF,
        "neutral_prev": NEUTRAL_PREVALENCE_SCORE, "neutral_age": NEUTRAL_AGE_SCORE,
        "site_buffer": SITE_BUFFER,
    })
    result = list(cursor)[0]
    similar_studies = result["similar_studies"]
    # Dedup near-duplicate facility records BEFORE taking the top N, so the cut is
    # applied to distinct physical sites (the query already returned a buffer of
    # SITE_BUFFER ranked rows to dedup against).
    sites = dedup_sites(result["recommended_sites"])
    print(f"found {len(similar_studies)} similar studies, {len(sites)} ranked sites")

    top_sites = sites[:top_n]

    print("retrieving dosage context (AutoGraph)...")
    dosage_lines, dosage_raw = get_dosage(similar_studies)
    dosage_source = "AutoGraph context graph"
    if not dosage_lines:
        # AutoGraph returned nothing parseable. Fall back to the matched trials'
        # own intervention text, which is deterministic and always present once
        # export_docs.py has run, so a transient or empty Retriever answer does
        # not leave the run with no dosage.
        fb = get_dosage_from_corpus(similar_studies)
        if fb:
            dosage_lines, dosage_source = fb, "trial intervention text (fallback)"
    dosage_on_topic = _dosage_on_topic(dosage_lines, similar_studies)
    print(f"  dosage: {('ok (' + dosage_source + ')') if dosage_lines else 'unavailable'}"
          + ("" if (not dosage_lines or dosage_on_topic) else " (off-topic for the matched trials)"))

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
        # Only whether dosage evidence exists, never the lines themselves. The
        # exact doses are printed in their own section; keeping them out of the
        # prompt makes it impossible for the model to restate or corrupt a dose or
        # a drug name.
        "dosage_available": bool(dosage_lines),
    }
    # Reasoning is explanatory, not load-bearing: if the model call fails, fall
    # back to the deterministic summary instead of crashing the run.
    try:
        reasoning = oai.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "system", "content": REASONING_PROMPT},
                      {"role": "user", "content": json.dumps(evidence)}],
        ).choices[0].message.content
    except Exception as e:
        print(f"  [reasoning call failed ({e}); using deterministic summary]")
        reasoning = _fallback_reasoning(top_sites)

    # The ranking is authoritative. Keep the model's reasoning unless it actually
    # contradicts the ranking (names a lower site as the overall winner, or omits
    # the top site); only then fall back to the deterministic summary.
    if not _reasoning_sound(reasoning, top_sites):
        print("  [reasoning contradicted the ranking; using deterministic summary]")
        reasoning = _fallback_reasoning(top_sites)

    confidence, confidence_notes = assess_confidence(
        similar_studies, sites, bool(dosage_lines), dosage_on_topic=dosage_on_topic)
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
        "dosage_lines": dosage_lines, "dosage_raw": dosage_raw, "dosage_source": dosage_source,
        "confidence": confidence, "confidence_notes": confidence_notes,
        "similar_studies": similar_studies, "stats": stats,
    }


if __name__ == "__main__":
    from arango import ArangoClient

    if len(sys.argv) < 2:
        print("usage: python src/resolver.py data/protocol_sample.json")
        sys.exit(1)

    client = ArangoClient(hosts=os.environ["ARANGO_HOST"])
    db = client.db(os.environ["ARANGO_DB"],
                   username=os.environ["ARANGO_USER"],
                   password=os.environ["ARANGO_PASSWORD"])
    # base_url defaults to OpenAI; set OPENAI_BASE_URL to use any
    # OpenAI-compatible provider for embeddings and reasoning.
    # An empty OPENAI_BASE_URL in .env would make the client use an empty base URL
    # and fail to connect, so treat empty as unset (defaults to OpenAI). The OpenAI
    # SDK reads OPENAI_BASE_URL from the environment when it is set.
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ.pop("OPENAI_BASE_URL", None)
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(sys.argv[1]) as f:
        protocol = json.load(f)

    result = resolve(db, oai, protocol)

    print("\nNote: the median-age signal is a synthetic stand-in (see the [synthetic] row below). "
          "This demonstrates the multimodel composition, not validated site rankings.")

    print("\n=== RECOMMENDED SITES (protocol + activity + prevalence + age) ===")
    for i, s in enumerate(result["top_sites"], 1):
        flag = "" if s.get("has_geo_data") else "  [missing geo data -> lower confidence]"
        print(f"\n{i}. {s['name']} -- {s['city']}, {s['country']}   overall: {round(s['composite_score'],3)}{flag}")
        mc = s.get("matched_count")
        if mc:
            best = s.get("best_similarity")
            # cosine can be slightly negative for a weak match; clamp the
            # displayed value to [0,1] so the printed "best match" never shows a
            # negative number.
            best_disp = None if best is None else max(0.0, min(1.0, best))
            print(f"   ran {mc} of the top matched studies"
                  + (f" (best match {round(best_disp, 3)})" if best_disp is not None else "")
                  + (f"; {s['trial_count']} active trials at site" if s.get("trial_count") is not None else ""))
        bd = score_breakdown(s)
        labels = {
            "protocol_similarity": "Protocol relevance",
            "site_activity":       "Site activity     ",
            "disease_prevalence":  "Disease prevalence",
            "age_match":           "Age match         ",
        }
        for key, (val, wt) in bd.items():
            # AGE_SYNTHETIC is a global flag: it is correct while every site's
            # median age is synthetic (enrich.py). If real ages are ever mixed in,
            # this label should switch to a per-site flag (facility.median_age_
            # synthetic) returned from the marquee, so only synthetic rows are tagged.
            tag = "  [synthetic]" if (key == "age_match" and AGE_SYNTHETIC) else ""
            print(f"   {labels[key]}  {val:<6}  ({int(wt*100)}%){tag}")

    print("\n=== EVIDENCE PASSAGES ===")
    for e in result["evidence_passages"]:
        print(f"- {e['study']} (similarity: {e['similarity']})")

    print(f"\n=== DOSAGE EVIDENCE ({result['dosage_source']}) ===")
    if result["dosage_lines"]:
        for line in result["dosage_lines"]:
            print(f"  - {line}")
        n = len(result["dosage_lines"])
        word = "statement" if n == 1 else "statements"
        print(f"  ({n} distinct dosage {word} extracted from {result['dosage_source']})")
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