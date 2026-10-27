"""
build_aact_sample.py — cut the vendored Breast Cancer sample out of a full AACT export.

The full AACT flat-file export (https://aact.ctti-clinicaltrials.org/downloads) is
~3.5 GB across ~590k studies. The pipeline only ever touches trials whose condition is
"Breast Cancer", so this script keeps every row of the eight tables the pipeline reads
whose nct_id belongs to one of those trials, and writes them to data/aact_sample/.

Every Breast Cancer trial is kept (any status), not only the active ones, so
ingest.load_studies() reports the same "found N Breast Cancer trials" count against the
sample as against the full export, and every downstream loader returns identical rows.

Place this file at:  scripts/build_aact_sample.py

Usage (only needed to build/refresh the sample from a full AACT snapshot):
    python scripts/build_aact_sample.py path/to/full_aact_export

where path/to/full_aact_export is the folder holding the unzipped AACT .txt files
(conditions.txt, studies.txt, ...). The output goes to data/aact_sample/ at the repo root.
"""

import os
import sys
import pandas as pd

# Output resolves relative to this file, so it always lands at <repo>/data/aact_sample
# regardless of where you run the script from. (This file is expected to live in scripts/.)
_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(_ROOT_DIR, "data", "aact_sample")

CONDITION_FILTER = "Breast Cancer"   # keep in sync with ingest.py / export_docs.py

# The tables ingest.py and export_docs.py read. Others in the export are not used.
TABLES = [
    "conditions.txt",
    "studies.txt",
    "brief_summaries.txt",
    "detailed_descriptions.txt",
    "eligibilities.txt",
    "facilities.txt",
    "facility_investigators.txt",
    "interventions.txt",
]

CHUNK = 200_000   # read big tables in chunks so a 3.5 GB export never loads fully into RAM


def _chunks(path):
    # na_filter=False keeps every value as the raw string, so the sample round-trips
    # byte-for-byte in meaning: the pipeline's own reader applies NaN handling later.
    return pd.read_csv(path, sep="|", dtype=str, na_filter=False,
                       low_memory=False, chunksize=CHUNK)


def breast_cancer_ids(src_dir):
    ids = set()
    for chunk in _chunks(os.path.join(src_dir, "conditions.txt")):
        ids.update(chunk.loc[chunk["name"].str.lower() == CONDITION_FILTER.lower(), "nct_id"])
    return ids


def main(src_dir):
    os.makedirs(OUT_DIR, exist_ok=True)
    ids = breast_cancer_ids(src_dir)
    print(f"{len(ids)} {CONDITION_FILTER} trials in {src_dir}")

    for name in TABLES:
        total = kept = 0
        out_path = os.path.join(OUT_DIR, name)
        header = True
        with open(out_path, "w", encoding="utf-8", newline="") as out:
            for chunk in _chunks(os.path.join(src_dir, name)):
                total += len(chunk)
                sub = chunk[chunk["nct_id"].isin(ids)]
                kept += len(sub)
                sub.to_csv(out, sep="|", index=False, header=header)
                header = False
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  {name:30s} {total:>9,} rows -> {kept:>7,} kept  ({size_mb:.1f} MB)")

    print(f"sample written to {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
