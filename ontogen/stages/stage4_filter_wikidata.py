"""
Stage 4 — Download & filter Wikidata properties (run ONCE).

Manual step:
    python stages/stage4_filter_wikidata.py

What it does:
  1. Fetches ALL Wikidata properties via SPARQL endpoint
  2. Filters to allowed datatypes (item, quantity, string, monolingualtext, time)
  3. Drops external-ID / external-KB-ID properties
  4. Converts P-IDs to PascalCase rdfs:label  (P19 → PlaceOfBirth)
  5. Saves to data/wikidata/properties_filtered.json

Output schema per entry:
  { "pid": "P19", "label": "PlaceOfBirth", "description": "..." }
"""
import json, re, sys, time
import requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ALLOWED_DATATYPES, DATABASE_URL

SPARQL_URL = "https://query.wikidata.org/sparql"

QUERY = """
SELECT ?prop ?propLabel ?propDescription ?datatype WHERE {
  ?prop a wikibase:Property ;
        wikibase:propertyType ?datatype .
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}
"""

EXTERNAL_ID_TYPES = {
    "wikibase-externalid",
    "external-id",
}


def pid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]  # "http://www.wikidata.org/entity/P19" → "P19"


def datatype_slug(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].lower().replace("-", "")


def to_pascal_case(label: str) -> str:
    "'place of birth' → 'PlaceOfBirth'"
    return "".join(w.capitalize() for w in re.split(r"[\s_\-]+", label) if w)


def fetch_all_properties() -> list[dict]:
    headers = {"Accept": "application/sparql-results+json", "User-Agent": "OntogenBot/1.0"}
    params  = {"query": QUERY, "format": "json"}

    print("[Stage 4] Querying Wikidata SPARQL … (may take 30–60 s)")
    print(f"[Stage 4]   URL: {SPARQL_URL}")
    print(f"[Stage 4]   Headers: {headers}")
    print(f"[Stage 4]   Query preview: {QUERY[:120].strip()} ...")

    try:
        r = requests.get(SPARQL_URL, params=params, headers=headers, timeout=120)
    except requests.exceptions.Timeout:
        print("[Stage 4] ✗ TIMEOUT — Wikidata did not respond within 120 s")
        print("[Stage 4]   Try again later or use the manual SPARQL download (see README).")
        raise
    except requests.exceptions.ConnectionError as e:
        print(f"[Stage 4] ✗ CONNECTION ERROR — cannot reach {SPARQL_URL}")
        print(f"[Stage 4]   Detail: {e}")
        print("[Stage 4]   Check: internet access? VPN? firewall?")
        raise

    print(f"[Stage 4]   HTTP status: {r.status_code}")
    print(f"[Stage 4]   Response headers: {dict(r.headers)}")

    if r.status_code == 429:
        retry = r.headers.get("Retry-After", "unknown")
        print(f"[Stage 4] ✗ RATE LIMITED (429) — Retry-After: {retry}s")
        print("[Stage 4]   Wait that long, then re-run. Or use the manual download.")
        r.raise_for_status()

    if r.status_code != 200:
        print(f"[Stage 4] ✗ Non-200 response. Body (first 500 chars):")
        print(r.text[:500])
        r.raise_for_status()

    print(f"[Stage 4]   Raw response size: {len(r.content):,} bytes")

    try:
        payload = r.json()
    except Exception as e:
        print(f"[Stage 4] ✗ JSON decode failed: {e}")
        print(f"[Stage 4]   Raw text (first 300 chars): {r.text[:300]}")
        raise

    if "results" not in payload:
        print(f"[Stage 4] ✗ 'results' key missing from response. Top-level keys: {list(payload.keys())}")
        raise KeyError("results")

    if "bindings" not in payload["results"]:
        print(f"[Stage 4] ✗ 'bindings' key missing. results keys: {list(payload['results'].keys())}")
        raise KeyError("bindings")

    bindings = payload["results"]["bindings"]
    print(f"[Stage 4]   → {len(bindings)} raw property bindings received")

    # Log a sample binding so we can see the shape
    if bindings:
        print(f"[Stage 4]   Sample binding[0]: {json.dumps(bindings[0], indent=2)}")

    return bindings


def filter_properties(bindings: list[dict]) -> list[dict]:
    results       = []
    seen_pids     = set()
    skipped_dtype = {}   # dtype → count
    skipped_extid = 0
    skipped_dupe  = 0
    missing_label = 0

    allowed_norm = {d.replace("-", "").replace("_", "") for d in ALLOWED_DATATYPES}
    print(f"[Stage 4] Filtering {len(bindings)} bindings …")
    print(f"[Stage 4]   Allowed normalised dtypes: {sorted(allowed_norm)}")

    for i, b in enumerate(bindings):
        uri       = b.get("prop", {}).get("value", "")
        label     = b.get("propLabel", {}).get("value", "")
        desc      = b.get("propDescription", {}).get("value", "")
        dtype_uri = b.get("datatype", {}).get("value", "")

        pid   = pid_from_uri(uri)
        dtype = datatype_slug(dtype_uri)

        # -- verbose every 500 rows so we can see progress without flooding
        if i % 500 == 0:
            print(f"[Stage 4]   ... processed {i}/{len(bindings)}  kept so far: {len(results)}")

        # skip external IDs
        if dtype in EXTERNAL_ID_TYPES:
            skipped_extid += 1
            continue

        dtype_norm = dtype.replace("-", "").replace("_", "")
        if dtype_norm not in allowed_norm:
            skipped_dtype[dtype_norm] = skipped_dtype.get(dtype_norm, 0) + 1
            continue

        if pid in seen_pids:
            skipped_dupe += 1
            continue
        seen_pids.add(pid)

        if not label:
            missing_label += 1

        pascal = to_pascal_case(label) if label else pid
        results.append({"pid": pid, "label": pascal, "description": desc})

    print(f"[Stage 4] Filter summary:")
    print(f"[Stage 4]   Kept:              {len(results)}")
    print(f"[Stage 4]   Skipped ext-ID:    {skipped_extid}")
    print(f"[Stage 4]   Skipped bad dtype: {sum(skipped_dtype.values())}  breakdown → {dict(sorted(skipped_dtype.items(), key=lambda x: -x[1])[:10])}")
    print(f"[Stage 4]   Skipped duplicates:{skipped_dupe}")
    print(f"[Stage 4]   Missing labels:    {missing_label}  (used P-ID as label)")

    if not results:
        print("[Stage 4] ✗ Zero properties survived filtering — something is wrong!")
        print("[Stage 4]   Check ALLOWED_DATATYPES in config.py and compare to sample binding above.")

    return results


def run():
    if not DATABASE_URL:
        print("[Stage 4] ✗ DATABASE_URL is not set — cannot write wikidata_properties.")
        sys.exit(1)

    bindings = fetch_all_properties()
    props    = filter_properties(bindings)

    # Upsert into wikidata_properties (embeddings filled by stage 5). This is a
    # one-time seed; for the initial migration prefer scripts/migrate_ontogen_files_to_db.py.
    from sqlalchemy import func, select
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from db.models import WikidataProperty
    from db.session import make_session_factory

    print(f"[Stage 4] Upserting {len(props)} properties into wikidata_properties …")
    factory = make_session_factory(DATABASE_URL)
    with factory() as session:
        for p in props:
            stmt = pg_insert(WikidataProperty).values(
                pid=p["pid"], label=p["label"], description=p.get("description", "")
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["pid"],
                set_={"label": stmt.excluded.label, "description": stmt.excluded.description},
            )
            session.execute(stmt)
        session.commit()
        total = session.execute(select(func.count()).select_from(WikidataProperty)).scalar_one()

    print(f"[Stage 4] ✓ wikidata_properties now holds {total} row(s). Done.")


if __name__ == "__main__":
    run()
