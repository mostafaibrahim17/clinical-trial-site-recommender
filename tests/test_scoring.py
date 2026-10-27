"""
test_scoring.py — unit tests for the deterministic scoring logic.

These cover the parts of resolver.py that do not need a live database or any
network call: the scoring weights, the margin-aware confidence downgrade, the
dosage-line parser, and the reasoning soundness guard. Run with:

    pytest test_scoring.py

resolver.py reads a few environment variables at import time, so dummy values
are set here before the import. They are never used by the functions under test.
"""

import os
import pytest

os.environ.setdefault("ARANGO_HOST", "https://example.invalid")
os.environ.setdefault("ARANGO_DB", "test")
os.environ.setdefault("ARANGO_USER", "test")
os.environ.setdefault("ARANGO_PASSWORD", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("RETRIEVER_HOST", "https://example.invalid")
os.environ.setdefault("RETRIEVER_SERVICE_ID", "arangodb-graphrag-retriever-test")

import resolver as R


# --- weights ---------------------------------------------------------------

def test_weights_sum_to_one():
    total = R.W_PROTOCOL + R.W_ACTIVITY + R.W_PREVALENCE + R.W_AGE
    assert abs(total - 1.0) < 1e-9


# --- margin-aware confidence ----------------------------------------------

def _sites(top, second):
    base = [{"composite_score": second, "has_geo_data": True} for _ in range(5)]
    base[0]["composite_score"] = top
    return base


def test_confidence_high_when_clear_and_complete():
    studies = [{"similarity": 0.82}, {"similarity": 0.78}]   # top >= MIN_TOP_SIMILARITY
    sites = _sites(0.90, 0.70)                               # clear 0.20 margin
    level, _ = R.assess_confidence(studies, sites, dosage_ok=True)
    assert level == "high"


def test_confidence_downgraded_on_near_tie():
    studies = [{"similarity": 0.82}, {"similarity": 0.78}]
    sites = _sites(0.847, 0.845)                             # within NEAR_TIE_EPS
    level, notes = R.assess_confidence(studies, sites, dosage_ok=True)
    assert level == "medium"
    assert "tied" in notes.lower()


def test_confidence_flags_off_topic_protocol():
    # Closest study below MIN_TOP_SIMILARITY: the protocol is off-topic and must
    # not be reported as high even with a clear site margin.
    studies = [{"similarity": 0.674}, {"similarity": 0.61}]
    sites = _sites(0.90, 0.70)
    level, notes = R.assess_confidence(studies, sites, dosage_ok=True)
    assert level != "high"
    assert "off-topic" in notes.lower()


def test_confidence_flags_off_topic_dosage():
    studies = [{"similarity": 0.82}, {"similarity": 0.78}]
    sites = _sites(0.90, 0.70)
    level, notes = R.assess_confidence(studies, sites, dosage_ok=True, dosage_on_topic=False)
    assert level != "high"
    assert "does not reference the matched trials" in notes.lower()


# --- dosage parsing --------------------------------------------------------

def test_dose_parse_keeps_hyphenated_drug_and_trims_eg():
    raw = (
        "* ARV-471 — 200 mg once daily, oral, for 24 weeks\n"
        "* Abemaciclib — 150 mg twice daily (e.g.\n"
        "* Palbociclib — no numeric dosing is given\n"
    )
    lines = R.summarize_dosage(raw)
    joined = " | ".join(lines)
    assert any(l.startswith("ARV-471:") for l in lines)      # in-name hyphen kept
    assert "Palbociclib" not in joined                        # no dose -> dropped
    assert "e.g" not in joined                                # dangling fragment trimmed


def test_dose_shape_requires_number_and_unit():
    assert R.summarize_dosage("* Ribociclib — once daily, no dose\n") == []


def test_dose_multi_drug_line_with_known_set_no_mislabel():
    # When one line lists several drugs and only the CDK4/6 class is known,
    # the non-CDK drugs' doses must be skipped, not relabelled as the CDK drug.
    # The earlier tests pass known_drugs=None, which hid this: a known set makes
    # letrozole/tamoxifen/goserelin implausible, so a naive backward scan falls
    # back to the leading "Palbociclib".
    cdk = {"palbociclib", "ribociclib", "abemaciclib", "dalpiciclib", "trilaciclib"}
    raw = ("Palbociclib 125 mg orally once daily, letrozole 2.5 mg PO daily, "
           "tamoxifen 20 mg PO daily, goserelin 3.6 mg SC")
    lines = R.summarize_dosage(raw, cdk)
    assert len(lines) == 1, lines
    assert lines[0].lower().startswith("palbociclib"), lines
    # None of the other drugs' doses may leak into the palbociclib line. 3.6 mg
    # alone passed even with the old 140-char window (it was trimmed as the
    # dangling tail); 2.5 mg and 20 mg sat mid-line and did leak, so assert them
    # all explicitly -- the window now ends at the next dose on the line.
    assert "2.5 mg" not in lines[0], lines    # letrozole dose must not leak over
    assert "20 mg" not in lines[0], lines     # tamoxifen dose must not leak over
    assert "3.6" not in lines[0], lines       # goserelin dose must not leak over


def test_dose_attribution_follows_the_preceding_drug():
    # PROPERTY (input-agnostic): on a line listing several drugs, each parsed line
    # must attribute the FIRST dose after a drug to THAT drug -- never to an earlier
    # one. Built from arbitrary (drug, dose) pairs; the assertion is about structure,
    # not any specific recommender output, so changing the drugs/order keeps it valid.
    pairs = [("Palbociclib", "125 mg"), ("Letrozole", "2.5 mg"),
             ("Tamoxifen", "20 mg"), ("Goserelin", "3.6 mg")]
    line = ", ".join(f"{d} {dose} PO daily" for d, dose in pairs)
    out = R.summarize_dosage(line, None)        # None -> every plausible drug kept
    emitted = {l.split(":", 1)[0].strip().lower(): l.lower() for l in out}
    for drug, dose in pairs:
        key = drug.lower()
        if key in emitted:                       # if this drug was parsed at all,
            # its line must begin with its OWN dose, not a neighbour's
            assert emitted[key].startswith(f"{key}: {dose.lower()}"), emitted[key]


# --- protocol input validation ---------------------------------------------

def test_validate_protocol_accepts_well_formed():
    p = {"title": "A phase III HR+/HER2- breast cancer trial",
         "description": "Palbociclib plus endocrine therapy in advanced disease.",
         "eligibility": "Postmenopausal women with HR+ HER2- advanced breast cancer."}
    R.validate_protocol(p)        # must not raise


def test_validate_protocol_rejects_missing_field():
    p = {"title": "x" * 30, "description": "y" * 30}   # no 'eligibility'
    with pytest.raises(ValueError):
        R.validate_protocol(p)


def test_validate_protocol_rejects_blank_field():
    p = {"title": "x" * 30, "description": "   ", "eligibility": "z" * 30}
    with pytest.raises(ValueError):
        R.validate_protocol(p)


def test_validate_protocol_rejects_non_dict():
    with pytest.raises(ValueError):
        R.validate_protocol("not a protocol")


# --- dosage on-topic check -------------------------------------------------

_MATCHED = [
    {"nct_id": "NCT02513394", "title": "PALLAS Palbociclib Collaborative Adjuvant Study"},
    {"nct_id": "NCT03155997", "title": "PENELOPE-B Palbociclib"},
]


def test_dosage_on_topic_true_when_drug_matches():
    # The check runs on parsed dose lines: True when a dosed drug appears in a
    # matched trial's title/summary.
    lines = ["Palbociclib: 125 mg once daily"]
    assert R._dosage_on_topic(lines, _MATCHED) is True


def test_dosage_off_topic_when_unrelated():
    # Doses for drugs not in the matched trials must read as off-topic, even
    # though the raw answer would echo the question's NCT ids.
    lines = ["Durvalumab: 1120 mg IV every 3 weeks", "Zanidatamab: 1800 mg"]
    assert R._dosage_on_topic(lines, _MATCHED) is False


# --- reasoning soundness ---------------------------------------------------

_TOP = [
    {"name": "Avera Cancer Institute-Aberdeen", "composite_score": 0.847},
    {"name": "Marin Cancer Center", "composite_score": 0.844},
    {"name": "CAMC Cancer Center", "composite_score": 0.844},
]


def test_reasoning_accepts_shortened_name_and_tradeoff():
    good = ("Avera leads overall. Marin has stronger local prevalence, but Avera "
            "makes up for it on site activity and finishes ahead.")
    assert R._reasoning_sound(good, _TOP) is True


def test_reasoning_rejects_inversion():
    bad = "Marin Cancer Center is the best overall choice; Avera is close behind."
    assert R._reasoning_sound(bad, _TOP) is False


def test_reasoning_rejects_missing_top_site():
    bad = "Marin and CAMC are both reasonable options to consider."
    assert R._reasoning_sound(bad, _TOP) is False