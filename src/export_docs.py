"""
export_docs.py — writes one .txt per study for AutoGraph ingestion.

Feeds the AutoGraph context graph (separate from the DB ingest). Each doc carries
Study ID + Summary + Eligibility + Interventions so AutoGraph can extract clinical
entities INCLUDING dosage (which lives in interventions.description, not a clean field).

Kept in sync with ingest.py on the filter + the therapeutic area.
The doc COUNT (MAX_DOCS) is capped for cost/time; keep this number consistent
with whatever the tutorial prose states.
"""

import os
import pandas as pd

DATA_DIR = "aact_sample"
OUT_DIR = "data/autograph_docs_50"     # folder name reflects the cap; keep in sync with MAX_DOCS

# --- kept in sync with ingest.py ---
CONDITION_FILTER = "Breast Cancer"
ACTIVE_STATUSES = ["ACTIVE_NOT_RECRUITING", "RECRUITING"]

# --- corpus size control ---
MAX_DOCS = 50          # start at 50; raise to 75/100 if AutoGraph coverage is thin
SAMPLE_SEED = 42       # reproducible sample


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
    ]["nct_id"].unique()

    # Only studies that actually have a summary (nothing to write otherwise)
    summaries = summaries[summaries["nct_id"].isin(active_ids)].dropna(subset=["description"])

    # Cap the corpus with a reproducible random sample
    if len(summaries) > MAX_DOCS:
        summaries = summaries.sample(n=MAX_DOCS, random_state=SAMPLE_SEED)
    print(f"exporting {len(summaries)} of {len(active_ids)} active studies (cap {MAX_DOCS})")

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
            desc = row.get("description", "") or ""
            lines.append(f"- ({itype}) {name}: {desc}".strip())
        intv_map[nct_id] = "\n".join(lines)

    os.makedirs(OUT_DIR, exist_ok=True)

    for _, row in summaries.iterrows():
        nct_id = row["nct_id"]
        elig_text = elig_map.get(nct_id, "")
        intv_text = intv_map.get(nct_id, "")

        with open(f"{OUT_DIR}/{nct_id}.txt", "w", encoding="utf-8") as f:
            f.write(f"Study: {nct_id}\n\n")
            f.write(f"Summary:\n{row['description']}\n\n")
            f.write(f"Eligibility Criteria:\n{elig_text}\n\n")
            f.write(f"Interventions:\n{intv_text}\n")   # dosage lives here -> lets AutoGraph extract it

    print(f"wrote {len(summaries)} documents to {OUT_DIR}/")


if __name__ == "__main__":
    main()