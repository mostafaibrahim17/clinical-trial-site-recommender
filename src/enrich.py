"""
enrich.py — adds external structured data onto facilities. FAST + RESUMABLE.

Runs AFTER ingest.py, BEFORE the resolver.

Design (see limitations_log.md):
  - Prevalence  : CDC PLACES all-cancer prevalence, county level, PROXY. REAL data.
  - Demographics: SYNTHETIC median age per county (seeded). Live Census ACS now needs a key (deferred).
  - Geo join    : facility lat/long -> county FIPS via keyless FCC Area API.
  - Non-US      : left null (confidence signal flags it).

Speed/resume:
  - Skips facilities already enriched (have county_fips) -> safe to re-run after an interrupt.
  - Collects UNIQUE coordinates first, geocodes each once, then writes facilities in a fast pass.
"""

from arango import ArangoClient
from dotenv import load_dotenv
import os
import time
import random
import requests

load_dotenv()

FCC_BLOCK_URL = "https://geo.fcc.gov/api/census/block/find"
PLACES_URL = "https://data.cdc.gov/resource/swc5-untb.json"
PLACES_CANCER_MEASURE = "CANCER"
SYNTH_AGE_SEED = 42
REQUEST_PAUSE = 0.05


def connect():
    client = ArangoClient(hosts=os.environ["ARANGO_HOST"])
    db = client.db(
        os.environ["ARANGO_DB"],
        username=os.environ["ARANGO_USER"],
        password=os.environ["ARANGO_PASSWORD"],
    )
    print("connected:", db.name)
    return db


def latlon_to_fips(lat, lon):
    try:
        r = requests.get(
            FCC_BLOCK_URL,
            params={"latitude": lat, "longitude": lon, "format": "json"},
            timeout=15,
        )
        r.raise_for_status()
        return (r.json().get("County") or {}).get("FIPS")
    except Exception as e:
        print(f"  geo lookup failed ({lat},{lon}): {e}")
        return None


def load_places_prevalence():
    print("pulling CDC PLACES cancer prevalence...")
    out = {}
    try:
        r = requests.get(
            PLACES_URL,
            params={"measureid": PLACES_CANCER_MEASURE,
                    "$select": "locationid,data_value", "$limit": 60000},
            timeout=60,
        )
        r.raise_for_status()
        for row in r.json():
            fips, val = row.get("locationid"), row.get("data_value")
            if fips and val is not None:
                out[str(fips).zfill(5)] = float(val)
    except Exception as e:
        print(f"  PLACES pull failed: {e}")
    print(f"  got prevalence for {len(out)} counties")
    return out


def synthetic_median_age(county_fips):
    rng = random.Random(f"{SYNTH_AGE_SEED}:{county_fips}")
    return round(rng.uniform(33.0, 47.0), 1)


def is_us(country):
    c = (country or "").strip().lower()
    return (not c) or c in ("united states", "usa", "us")


def enrich_facilities(db, prevalence):
    col = db.collection("facilities")

    # 1) Pull the facilities that still need enrichment (resumable: skip those already done)
    print("scanning facilities...")
    todo = []               # (key, lat, lon)
    unique_coords = set()
    skipped_done = skipped_non_us = skipped_no_geo = 0
    for fac in col.all():
        if fac.get("county_fips"):         # already enriched -> skip (resume)
            skipped_done += 1
            continue
        if not is_us(fac.get("country")):
            skipped_non_us += 1
            continue
        lat, lon = fac.get("latitude"), fac.get("longitude")
        if lat is None or lon is None:
            skipped_no_geo += 1
            continue
        todo.append((fac["_key"], lat, lon))
        unique_coords.add((lat, lon))

    print(f"to enrich: {len(todo)} facilities | {len(unique_coords)} unique coordinates")
    print(f"skipped: already-done {skipped_done}, non-US {skipped_non_us}, no-geo {skipped_no_geo}")

    # 2) Geocode each UNIQUE coordinate once
    coord_fips = {}
    for i, (lat, lon) in enumerate(unique_coords, 1):
        coord_fips[(lat, lon)] = latlon_to_fips(lat, lon)
        time.sleep(REQUEST_PAUSE)
        if i % 200 == 0:
            print(f"  geocoded {i}/{len(unique_coords)} unique coords")

    # 3) Write facilities in a fast pass
    enriched = 0
    for key, lat, lon in todo:
        county_fips = coord_fips.get((lat, lon))
        if not county_fips:
            continue
        update = {"_key": key, "county_fips": county_fips,
                  "median_age": synthetic_median_age(county_fips),
                  "median_age_synthetic": True}
        if county_fips in prevalence:
            update["cancer_prevalence"] = prevalence[county_fips]
        col.update(update)
        enriched += 1

    print(f"enriched this run: {enriched}")


if __name__ == "__main__":
    db = connect()
    prevalence = load_places_prevalence()
    enrich_facilities(db, prevalence)
    print("enrich complete")