from arango import ArangoClient
from dotenv import load_dotenv
from openai import OpenAI
import os
import sys
import pandas as pd
import random
import json
import math

load_dotenv()

# --- Constants ---
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CONDITION_FILTER = "Breast Cancer"          # shared filter; keep export_docs.py in sync with this
ACTIVE_STATUSES = ["ACTIVE_NOT_RECRUITING", "RECRUITING"]  # active + currently-recruiting sites
STUDY_TYPE = "INTERVENTIONAL"               # treatment trials only; excludes observational studies

# Number of inverted lists for the studies vector index. The active+recruiting
# breast-cancer set is small (low hundreds), so a small nLists trains a stable
# quantizer; raise it only if the study corpus grows substantially.
VECTOR_NLISTS = 2

# Paths resolved relative to this file so the notebook can import from anywhere
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SRC_DIR, "..")
DATA_DIR = os.path.join(_ROOT_DIR, "data", "aact_sample")
PERF_FILE = os.path.join(_ROOT_DIR, "data", "synthetic_performance.json")


# --- Connection ---
def connect():
    client = ArangoClient(hosts=os.environ["ARANGO_HOST"])
    db = client.db(
        os.environ["ARANGO_DB"],
        username=os.environ["ARANGO_USER"],
        password=os.environ["ARANGO_PASSWORD"],
    )
    print("connected:", db.name)
    return db


# --- Helpers ---
def clean(doc):
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in doc.items()}


def embed(oai, texts):
    response = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in response.data]


def _read(name):
    """Read a pipe-delimited AACT file into a DataFrame.
    dtype=str avoids pandas mis-inferring int columns (AACT has messy
    numeric fields like enrollment that break int parsing)."""
    return pd.read_csv(
        os.path.join(DATA_DIR, name), sep="|", low_memory=False, dtype=str
    )


# --- Idempotence ---
def already_ingested(db):
    """Return True when the database already holds a built study corpus with its
    vector index, so a plain run can safely skip the load and the paid embedding
    pass. A run can still force a clean rebuild with the --rebuild flag."""
    if not db.has_collection("studies"):
        return False
    studies = db.collection("studies")
    if studies.count() == 0:
        return False
    has_index = any(idx.get("name") == "studies_vec" for idx in studies.indexes())
    if not has_index:
        return False
    embedded = list(db.aql.execute(
        "FOR s IN studies FILTER s.embedding != null LIMIT 1 RETURN 1"
    ))
    return bool(embedded)


# --- Schema ---
def create_schema(db):
    for name in ["studies", "facilities", "investigators"]:
        if not db.has_collection(name):
            db.create_collection(name)
            print(f"created collection: {name}")

    if not db.has_collection("site_relationships"):
        db.create_collection("site_relationships", edge=True)
        print("created edge collection: site_relationships")

    if not db.has_graph("site_topology"):
        graph = db.create_graph("site_topology")
        graph.create_edge_definition(
            edge_collection="site_relationships",
            from_vertex_collections=["facilities", "investigators", "studies"],
            to_vertex_collections=["facilities", "investigators", "studies"],
        )
        print("created graph: site_topology")

    # Index the facility identity fields so the resolver's per-hospital trial-count
    # subquery (filter on name+city+state+country) does not scan the whole collection.
    # Guard on the index NAME so a re-run against an already-built DB skips cleanly
    # instead of failing with a duplicate-name error (ERR 1005).
    fac = db.collection("facilities")
    if "facility_identity" not in {ix.get("name") for ix in fac.indexes()}:
        fac.add_index({"type": "persistent", "fields": ["name", "city", "state", "country"],
                       "name": "facility_identity", "sparse": False})
        print("created index: facilities.facility_identity")


# --- Load AACT Data ---
def load_studies():
    conditions = _read("conditions.txt")
    all_bc_nct_ids = conditions[
        conditions["name"].str.lower() == CONDITION_FILTER.lower()
    ]["nct_id"].unique()
    print(f"found {len(all_bc_nct_ids)} {CONDITION_FILTER} trials")

    studies = _read("studies.txt")
    studies = studies[
        (studies["nct_id"].isin(all_bc_nct_ids))
        & (studies["overall_status"].isin(ACTIVE_STATUSES))
        & (studies["study_type"].str.upper() == STUDY_TYPE)   # treatment trials only
    ][
        ["nct_id", "brief_title", "overall_status", "phase", "enrollment", "study_type"]
    ].dropna(subset=["brief_title"])
    print(f"loaded {len(studies)} active+recruiting interventional studies")

    return studies


def load_facilities_and_investigators(active_nct_ids):
    facilities = _read("facilities.txt")
    facilities = facilities[facilities["nct_id"].isin(active_nct_ids)][
        ["id", "nct_id", "name", "city", "state", "country", "latitude", "longitude"]
    ].dropna(subset=["name"])
    print(f"loaded {len(facilities)} facilities")

    # Filter investigators on the same active_nct_ids as everything else so the
    # investigator set matches the studies and facilities in scope.
    investigators = _read("facility_investigators.txt")
    investigators = investigators[investigators["nct_id"].isin(active_nct_ids)][
        ["id", "nct_id", "facility_id", "role", "name"]
    ].dropna(subset=["name"])
    print(f"loaded {len(investigators)} investigators")

    return facilities, investigators


def load_free_text(active_nct_ids):
    summaries = _read("brief_summaries.txt")
    summaries = summaries[summaries["nct_id"].isin(active_nct_ids)][
        ["nct_id", "description"]
    ].dropna(subset=["description"])
    print(f"loaded {len(summaries)} brief summaries")

    eligibilities = _read("eligibilities.txt")
    eligibilities = eligibilities[eligibilities["nct_id"].isin(active_nct_ids)][
        ["nct_id", "criteria", "minimum_age", "maximum_age", "gender"]
    ].dropna(subset=["criteria"])
    print(f"loaded {len(eligibilities)} eligibility records")

    # detailed_descriptions: the long-form protocol text, folded into embed_text
    # for richer semantic matches.
    detailed = _read("detailed_descriptions.txt")
    detailed = detailed[detailed["nct_id"].isin(active_nct_ids)][
        ["nct_id", "description"]
    ].dropna(subset=["description"])
    print(f"loaded {len(detailed)} detailed descriptions")

    return summaries, eligibilities, detailed


def load_interventions(active_nct_ids):
    """Staged for the dosage variable. Loaded now so it lives in the DB;
    dosage extraction happens later via AutoGraph, not here."""
    interventions = _read("interventions.txt")
    interventions = interventions[interventions["nct_id"].isin(active_nct_ids)][
        ["id", "nct_id", "intervention_type", "name", "description"]
    ].dropna(subset=["name"])
    print(f"loaded {len(interventions)} interventions")
    return interventions


# --- Synthetic Performance Data ---
def generate_synthetic_performance(facilities):
    """Generate stand-in site performance signals (illustrative only).

    Site-level trial outcomes are not public, so these fields are synthetic and
    every facility is tagged with performance_synthetic=True. They are kept as
    illustrative metadata and are NOT used by the recommender: the ranking's
    site-activity signal is a real count of active+recruiting interventional
    trials per site, computed in the marquee query (see resolver.MARQUEE). A real
    operational feed would replace these fields.
    """
    random.seed(42)
    performance = {}
    for facility_id in facilities["id"].unique():
        performance[str(facility_id)] = {
            "enrollment_rate": round(random.uniform(0.4, 1.0), 2),
            "activation_on_time": random.choice([True, True, True, False]),
            "success_score": round(random.uniform(0.5, 1.0), 2),
            "past_trials_count": random.randint(1, 20),
            "performance_synthetic": True,
        }
    print(f"generated synthetic performance for {len(performance)} facilities")

    with open(PERF_FILE, "w") as f:
        json.dump(performance, f)

    return performance


# --- Ingest into Arango ---
def ingest_studies(db, studies, summaries, eligibilities, detailed):
    col = db.collection("studies")
    col.truncate()

    summary_map = summaries.set_index("nct_id")["description"].to_dict()
    eligibility_map = eligibilities.set_index("nct_id")["criteria"].to_dict()
    detailed_map = detailed.set_index("nct_id")["description"].to_dict()

    docs = []
    for _, row in studies.iterrows():
        doc = clean(row.to_dict())
        if not doc.get("nct_id"):
            continue   # skip rows with a missing NCT id: no valid key to store under
        doc["_key"] = doc["nct_id"]
        doc["summary"] = summary_map.get(doc["nct_id"], "")
        doc["eligibility_criteria"] = eligibility_map.get(doc["nct_id"], "")
        doc["detailed_description"] = detailed_map.get(doc["nct_id"], "")
        # embed_text includes detailed_description for richer semantic matches
        doc["embed_text"] = (
            f"{doc['brief_title']} {doc['summary']} "
            f"{doc['eligibility_criteria']} {doc['detailed_description']}"
        )
        docs.append(doc)

    col.insert_many(docs)
    print(f"ingested {len(docs)} studies")
    return docs


def ingest_facilities(db, facilities, performance):
    col = db.collection("facilities")
    col.truncate()

    docs = []
    for _, row in facilities.iterrows():
        doc = clean(row.to_dict())
        doc["_key"] = str(doc["id"])
        perf = performance.get(str(doc["id"]), {})
        doc.update(perf)
        docs.append(doc)

    col.insert_many(docs)
    print(f"ingested {len(docs)} facilities")


def ingest_investigators(db, investigators):
    col = db.collection("investigators")
    col.truncate()

    docs = []
    for _, row in investigators.iterrows():
        doc = clean(row.to_dict())
        doc["_key"] = str(doc["id"])
        docs.append(doc)

    col.insert_many(docs)
    print(f"ingested {len(docs)} investigators")


def ingest_edges(db, facilities, investigators):
    col = db.collection("site_relationships")
    col.truncate()

    edges = []
    for _, row in facilities.iterrows():
        edges.append({
            "_from": f"studies/{row['nct_id']}",
            "_to": f"facilities/{row['id']}",
            "type": "ran_at",
        })

    for _, row in investigators.iterrows():
        edges.append({
            "_from": f"facilities/{row['facility_id']}",
            "_to": f"investigators/{row['id']}",
            "type": "has_investigator",
        })

    col.insert_many(edges)
    print(f"ingested {len(edges)} edges")


# --- Embeddings ---
def embed_and_index(db, oai, docs):
    col = db.collection("studies")

    # Skip if the vector index already exists and the studies still carry
    # embeddings, so calling this a second time (e.g. re-running the notebook's
    # embed cell) does not re-charge the embedding API. A clean rebuild truncates
    # the studies first, so this guard is not hit on the --rebuild path.
    if any(idx.get("name") == "studies_vec" for idx in col.indexes()):
        if list(db.aql.execute(
                "FOR s IN studies FILTER s.embedding != null LIMIT 1 RETURN 1")):
            print("vector index and embeddings already exist, skipping")
            return

    print("embedding studies...")
    texts = [d["embed_text"] for d in docs]

    all_vectors = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        vectors = embed(oai, batch)
        all_vectors.extend(vectors)
        print(f"  embedded {min(i + 100, len(texts))}/{len(texts)}")

    # Attach each embedding to its document and upsert in one pass. Using
    # insert_many(overwrite=True) replaces the existing study documents by key
    # and inserts any that are missing, so it never fails with "document not
    # found" the way a per-key update does when a row carried a bad key.
    for doc, vector in zip(docs, all_vectors):
        doc["embedding"] = vector
    col.insert_many(docs, overwrite=True)
    print("embeddings stored")

    # Rebuild the vector index cleanly: drop any existing one first so a rebuild
    # never leaves a stale index behind, then recreate it. Confirm the vector-
    # index syntax and nLists value against the target ArangoDB release; nLists
    # is kept small (VECTOR_NLISTS) to match the size of the study corpus.
    for idx in col.indexes():
        if idx.get("name") == "studies_vec":
            col.delete_index(idx["id"])
    col.add_index({
        "name": "studies_vec",
        "type": "vector",
        "fields": ["embedding"],
        "params": {"metric": "cosine", "dimension": EMBED_DIM, "nLists": VECTOR_NLISTS},
    })
    print("vector index created")


def build(db, oai):
    """Load every collection and build the vector index from scratch."""
    studies = load_studies()
    active_nct_ids = studies["nct_id"].unique()

    facilities, investigators = load_facilities_and_investigators(active_nct_ids)
    summaries, eligibilities, detailed = load_free_text(active_nct_ids)
    load_interventions(active_nct_ids)  # staged for dosage (AutoGraph later)

    performance = generate_synthetic_performance(facilities)

    docs = ingest_studies(db, studies, summaries, eligibilities, detailed)
    ingest_facilities(db, facilities, performance)
    ingest_investigators(db, investigators)
    ingest_edges(db, facilities, investigators)
    embed_and_index(db, oai, docs)


# --- Run as script ---
if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv[1:]

    db = connect()
    create_schema(db)
    print("schema ready")

    if already_ingested(db) and not rebuild:
        print("studies corpus and vector index already present; skipping ingest. "
              "Pass --rebuild to reload and re-embed from scratch.")
        sys.exit(0)

    # base_url defaults to OpenAI; set OPENAI_BASE_URL to use any
    # OpenAI-compatible provider (embeddings run against the same endpoint).
    # An empty OPENAI_BASE_URL in .env would make the client use an empty base URL
    # and fail to connect, so treat empty as unset (defaults to OpenAI). The OpenAI
    # SDK reads OPENAI_BASE_URL from the environment when it is set.
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ.pop("OPENAI_BASE_URL", None)
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    build(db, oai)
    print("ingest complete")