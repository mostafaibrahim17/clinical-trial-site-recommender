"""
export_docs.py — writes one .txt per study for AutoGraph ingestion.

Feeds the AutoGraph context graph (separate from the DB ingest). Each doc carries
Study ID + Summary + Eligibility + Interventions so AutoGraph can extract clinical
entities INCLUDING dosage (which lives in interventions.description, not a clean field).

The corpus is built from the trials in the protocol's DRUG CLASS (DRUG_KEYWORDS),
not an arbitrary sample, so the Retriever's dosage answers reference the drug class
the recommender actually matches on. Set DRUG_KEYWORDS to your protocol's class and
rebuild the AutoGraph graph when it changes. MAX_DOCS is a safety cap only; the
output folder name is derived from it.
"""

import os
import pandas as pd

DATA_DIR = "data/aact_sample"

# --- kept in sync with ingest.py ---
CONDITION_FILTER = "Breast Cancer"
ACTIVE_STATUSES = ["ACTIVE_NOT_RECRUITING", "RECRUITING"]
STUDY_TYPE = "INTERVENTIONAL"   # treatment trials only; keep in sync with ingest.py

# --- drug-class corpus selection ---
# Trials whose interventions mention one of these drugs form the dosage corpus.
# The default is the CDK4/6-inhibitor class the sample palbociclib protocol matches
# on; change it to your protocol's drug class and rebuild the AutoGraph graph.
# Set to [] to fall back to all active interventional trials in the area.
DRUG_KEYWORDS = [
    "palbociclib", "ribociclib", "abemaciclib", "dalpiciclib", "trilaciclib",
]

# --- corpus size control ---
# The drug-class set is exported in full by default: capping it with a random
# sample can drop trials that the recommender actually matches on, which then
# makes their dosage unretrievable even though the drug is in class. Set MAX_DOCS
# to a positive integer only as a hard ceiling for a very large class; None keeps
# every drug-class trial.
MAX_DOCS = None        # None = export the full drug-class set (no cap)
SAMPLE_SEED = 42       # reproducible sample, used only if MAX_DOCS is set

# Output folder; when uncapped it reflects the full drug-class corpus.
OUT_DIR = "data/autograph_docs" if MAX_DOCS is None else f"data/autograph_docs_{MAX_DOCS}"


def _read(name):
    # dtype=str avoids pandas mis-inferring messy AACT numeric columns
    return pd.read_csv(f"{DATA_DIR}/{name}", sep="|", low_memory=False, dtype=str)


def main():
    conditions = _read("conditions.txt")
    studies = _read("studies.txt")
    summaries = _read("brief_summaries.txt")
    eligibilities = _read("eligibilities.txt")
    interventions = _read("interventions.txt")

    # Filter to active studies in the target therapeutic area (same as ingest.py)
    bc_nct_ids = conditions[
        conditions["name"].str.lower() == CONDITION_FILTER.lower()
    ]["nct_id"].unique()
    active_ids = studies[
        (studies["nct_id"].isin(bc_nct_ids))
        & (studies["overall_status"].isin(ACTIVE_STATUSES))
        & (studies["study_type"].str.upper() == STUDY_TYPE)
    ]["nct_id"].unique()

    # Restrict to the protocol's drug class: trials whose interventions name one of
    # DRUG_KEYWORDS. This makes the dosage corpus on-topic by construction instead
    # of relying on a random sample to happen to include the right trials.
    if DRUG_KEYWORDS:
        intv = interventions[interventions["nct_id"].isin(active_ids)].copy()
        blob = (intv["name"].fillna("") + " " + intv.get("description", "").fillna("")).str.lower()
        pattern = "|".join(DRUG_KEYWORDS)
        class_ids = intv[blob.str.contains(pattern, regex=True)]["nct_id"].unique()
        active_ids = class_ids
        print(f"drug-class corpus: {len(active_ids)} trials matching {DRUG_KEYWORDS}")

    # Only studies that actually have a summary (nothing to write otherwise)
    summaries = summaries[summaries["nct_id"].isin(active_ids)].dropna(subset=["description"])

    # Optional hard ceiling (reproducible) only if MAX_DOCS is set. Off by default
    # so no matched trial is silently dropped from the corpus.
    if MAX_DOCS is not None and len(summaries) > MAX_DOCS:
        summaries = summaries.sample(n=MAX_DOCS, random_state=SAMPLE_SEED)
        print(f"exporting {len(summaries)} of {len(active_ids)} drug-class studies (cap {MAX_DOCS})")
    else:
        print(f"exporting all {len(summaries)} drug-class studies (uncapped)")

    # Lookups
    elig_map = (
        eligibilities[eligibilities["nct_id"].isin(active_ids)]
        .set_index("nct_id")["criteria"]
        .to_dict()
    )
    # interventions: multiple rows per study -> group the descriptions together
    intv = interventions[interventions["nct_id"].isin(active_ids)].dropna(subset=["name"])
    intv_map = {}
    for nct_id, grp in intv.groupby("nct_id"):
        lines = []
        for _, row in grp.iterrows():
            name = row.get("name", "")
            itype = row.get("intervention_type", "")
            desc = row.get("description", "")
            # Skip missing/blank descriptions instead of writing a literal "nan"
            # (dtype=str leaves missing values as NaN, which stringifies to "nan").
            desc = "" if pd.isna(desc) else str(desc).strip()
            if desc and desc.lower() != "nan":
                lines.append(f"- ({itype}) {name}: {desc}".strip())
            else:
                lines.append(f"- ({itype}) {name}".strip())
        intv_map[nct_id] = "\n".join(lines)

    os.makedirs(OUT_DIR, exist_ok=True)

    for _, row in summaries.iterrows():
        nct_id = row["nct_id"]
        elig_text = elig_map.get(nct_id, "")
        # elig_map can hold a NaN value (missing criteria), which the .get default
        # does not catch, so guard it the same way as the interventions above.
        elig_text = "" if pd.isna(elig_text) else str(elig_text).strip()
        if elig_text.lower() == "nan":
            elig_text = ""
        intv_text = intv_map.get(nct_id, "")

        with open(f"{OUT_DIR}/{nct_id}.txt", "w", encoding="utf-8") as f:
            f.write(f"Study: {nct_id}\n\n")
            f.write(f"Summary:\n{row['description']}\n\n")
            f.write(f"Eligibility Criteria:\n{elig_text}\n\n")
            f.write(f"Interventions:\n{intv_text}\n")   # dosage lives here -> lets AutoGraph extract it

    print(f"wrote {len(summaries)} documents to {OUT_DIR}/")


if __name__ == "__main__":
    main()