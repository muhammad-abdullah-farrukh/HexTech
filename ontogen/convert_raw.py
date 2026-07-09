# convert_raw.py — drop in ontogen/ and run once
import json, re
from pathlib import Path
from config import WIKIDATA_FILTERED, ALLOWED_DATATYPES

raw = json.loads(Path("wikidata_raw.json").read_text())

# Wikidata's "Download → JSON" export is a flat list of dicts with plain
# string values (no results.bindings wrapper, no .value nesting).
# But the raw SPARQL endpoint format IS wrapped that way. Handle both.
if isinstance(raw, dict) and "results" in raw:
    bindings = raw["results"]["bindings"]
    nested = True
else:
    bindings = raw
    nested = False

def pid_from_uri(uri): return uri.rsplit("/", 1)[-1]
def dtype_slug(uri): return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1].lower().replace("-", "")
def pascal(s): return "".join(w.capitalize() for w in re.split(r"[\s_\-]+", s) if w)

def field(b, key):
    v = b.get(key, "")
    if nested:
        return v.get("value", "") if isinstance(v, dict) else ""
    return v if isinstance(v, str) else ""

allowed = {d.replace("-", "").replace("_", "") for d in ALLOWED_DATATYPES}
seen, results = set(), []

for b in bindings:
    pid   = pid_from_uri(field(b, "prop"))
    label = field(b, "propLabel")
    desc  = field(b, "propDescription")
    dtype = dtype_slug(field(b, "datatype"))

    if dtype in {"wikibaseexternalid", "externalid"}:
        continue
    if dtype.replace("-", "").replace("_", "") not in allowed:
        continue
    if pid in seen:
        continue
    seen.add(pid)
    results.append({"pid": pid, "label": pascal(label) if label else pid, "description": desc})

WIKIDATA_FILTERED.parent.mkdir(parents=True, exist_ok=True)
WIKIDATA_FILTERED.write_text(json.dumps(results, indent=2))
print(f"Saved {len(results)} properties → {WIKIDATA_FILTERED}")