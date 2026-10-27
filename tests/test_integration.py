"""
test_integration.py — property tests against a live, built database.

These assert that the recommender behaves CORRECTLY, never that it produces a
particular answer. There are no expected site names, NCT ids, or scores anywhere
here: change data/protocol_sample.json or the AACT data and every test below still
passes, because each one checks an invariant that must hold for ANY input.

They need a built + enriched database and an OpenAI key, read from .env exactly as
resolver.py reads them. If those are not set the whole module is skipped, so the
offline suite (test_scoring.py) still runs anywhere.

Run from the repo root:
    # PowerShell:  $env:PYTHONPATH="src"; python -m pytest tests -q
    # bash:        PYTHONPATH=src python -m pytest tests -q
"""
import os
import json
import pytest
from dotenv import load_dotenv

load_dotenv()

# Decide up front whether we can reach a live stack; skip the module cleanly if not.
_REQUIRED = ["ARANGO_HOST", "ARANGO_DB", "ARANGO_USER", "ARANGO_PASSWORD", "OPENAI_API_KEY"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=f"integration env not set ({', '.join(_missing)}); offline tests still run",
)

# resolver.py reads a couple of Retriever vars at import time; give them harmless
# defaults so importing it never fails. These tests never call the Retriever.
os.environ.setdefault("RETRIEVER_HOST", os.environ.get("ARANGO_HOST", "https://example.invalid"))
os.environ.setdefault("RETRIEVER_SERVICE_ID", "integration-test")

# An empty OPENAI_BASE_URL in .env would make the OpenAI client use an empty base
# URL and fail to connect, so treat empty as unset (defaults to OpenAI) -- the same
# guard resolver.py and ingest.py apply before building their clients.
if not os.environ.get("OPENAI_BASE_URL"):
    os.environ.pop("OPENAI_BASE_URL", None)

import resolver as R   # noqa: E402  (import after the env defaults above)


# --- shared fixtures (built once per module) -------------------------------

@pytest.fixture(scope="module")
def db():
    from arango import ArangoClient
    client = ArangoClient(hosts=os.environ["ARANGO_HOST"])
    conn = client.db(os.environ["ARANGO_DB"],
                     username=os.environ["ARANGO_USER"],
                     password=os.environ["ARANGO_PASSWORD"])
    if conn.collection("studies").count() == 0:
        pytest.skip("studies collection is empty; build the database first")
    return conn


@pytest.fixture(scope="module")
def protocol_vec():
    """Embed the sample protocol exactly as resolver.resolve() does."""
    from openai import OpenAI
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "data", "protocol_sample.json")) as f:
        p = json.load(f)
    text = f"{p['title']} {p['description']} {p['eligibility']}"
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return oai.embeddings.create(model=R.EMBED_MODEL, input=[text]).data[0].embedding


def _marquee_bind(vec):
    """The exact bind parameters resolver.resolve() passes to the marquee query,
    pulled from resolver's own constants so the test can't drift from the code."""
    return {
        "vec": vec, "depth": R.TRAVERSAL_DEPTH,
        "w_protocol": R.W_PROTOCOL, "w_activity": R.W_ACTIVITY,
        "w_prev": R.W_PREVALENCE, "w_age": R.W_AGE,
        "prev_ref": R.PREVALENCE_REF, "activity_ref": R.ACTIVITY_REF,
        "neutral_prev": R.NEUTRAL_PREVALENCE_SCORE, "neutral_age": R.NEUTRAL_AGE_SCORE,
        "site_buffer": R.SITE_BUFFER,
    }


def _run_marquee(db, vec):
    return list(db.aql.execute(R.MARQUEE, bind_vars=_marquee_bind(vec)))[0]


# --- the invariants --------------------------------------------------------

def test_vector_search_returns_the_true_nearest_neighbours(db, protocol_vec):
    """The approximate index must return the SAME studies as an exhaustive exact
    cosine scan. This is the recall guarantee: it fails the moment APPROX_NEAR_COSINE
    under-returns (e.g. nProbe < VECTOR_NLISTS), with no hardcoded expected result."""
    approx = _run_marquee(db, protocol_vec)["similar_studies"]
    k = len(approx)
    assert k > 0, "vector search returned nothing -- is the DB built and embedded?"

    exact = list(db.aql.execute(
        "FOR s IN studies FILTER s.embedding != null "
        "  LET sim = COSINE_SIMILARITY(s.embedding, @vec) "
        "  SORT sim DESC LIMIT @k "
        "  RETURN { nct_id: s.nct_id, sim: sim }",
        bind_vars={"vec": protocol_vec, "k": k},
    ))

    assert {a["nct_id"] for a in approx} == {e["nct_id"] for e in exact}, (
        "APPROX_NEAR_COSINE missed true nearest neighbours -- vector index recall "
        "problem (check nProbe vs ingest.VECTOR_NLISTS)."
    )
    exact_by_id = {e["nct_id"]: e["sim"] for e in exact}
    for a in approx:
        assert abs(a["similarity"] - exact_by_id[a["nct_id"]]) < 1e-4


def test_ranking_is_deterministic(db, protocol_vec):
    """Same input -> same ranking. No randomness may leak into the ordering."""
    def ranking():
        rows = _run_marquee(db, protocol_vec)["recommended_sites"]
        return [(r["name"], r["city"], round(r["composite_score"], 9)) for r in rows]
    assert ranking() == ranking()


def test_composite_is_the_weighted_sum_of_its_subscores(db, protocol_vec):
    """Each site's composite must equal protocol*Wp + activity*Wa + prevalence*Wprev
    + age*Wage. Guards against a stray term or a weight/column mismatch."""
    rows = _run_marquee(db, protocol_vec)["recommended_sites"]
    assert rows, "no sites ranked"
    for s in rows:
        expected = (s["matched_similarity"] * R.W_PROTOCOL
                    + s["activity_score"]   * R.W_ACTIVITY
                    + s["prevalence_score"] * R.W_PREVALENCE
                    + s["age_score"]        * R.W_AGE)
        assert abs(s["composite_score"] - expected) < 1e-6, (
            s["name"], s["composite_score"], expected)


def test_subscores_are_bounded_and_similar_studies_sorted(db, protocol_vec):
    """Every sub-score sits in [0, 1] (the clamps hold) and the matched studies come
    back sorted by descending similarity."""
    res = _run_marquee(db, protocol_vec)
    sims = [s["similarity"] for s in res["similar_studies"]]
    assert sims == sorted(sims, reverse=True)
    for s in res["recommended_sites"]:
        for key in ("matched_similarity", "activity_score", "prevalence_score", "age_score"):
            assert 0.0 <= s[key] <= 1.0, (s["name"], key, s[key])