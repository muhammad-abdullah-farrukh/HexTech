# ontogen

LLM-driven ontology + KG construction pipeline, implemented verbatim from the paper.

## Setup

Dependencies for the whole HexTech project (parser + Ontogen) are managed in one
pinned file at the repo root. From the `HexTech/` root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --no-deps -r requirements.txt   # --no-deps: see requirements.txt header
```

The LLM endpoint/model are configured in `ontogen/config.py` (and the parser's
`.env`); no `OPENAI_API_KEY` is needed for the local server.

## One-time setup (Wikidata)

```bash
# 1. Download + filter Wikidata properties (~30-60s, needs internet)
python stages/stage4_filter_wikidata.py

# 2. Embed filtered properties with bge-small-en (~5-10min, downloads model on first run)
python stages/stage5_embed_wikidata.py
```

These produce:
- `data/wikidata/properties_filtered.json`
- `embeddings/wikidata_embeddings.npy`

## Run pipeline

```bash
# Drop .txt files into data/documents/
cp mydoc.txt data/documents/

# Run all docs
python pipeline.py

# Or single doc
python pipeline.py data/documents/mydoc.txt
```

Outputs land in `outputs/{cqs,answers,relations,ontology,kg}/`.

## Config

Edit `config.py`:
- `LLM_MODEL` — swap to any OpenAI-compatible model
- `SCHEMA_EXPANSION` — `True` = no-schema-constraint mode, `False` = target-schema-constrained
- `EMBED_MODEL` — paper uses `BAAI/bge-small-en` (don't change unless replicating a variant)

## Stage map

| Stage | File | Notes |
|-------|------|-------|
| 1 | `stages/stage1_cq_gen.py` | CQ generation |
| 2 | `stages/stage2_cq_answer.py` | QA per CQ |
| 3 | `stages/stage3_relation_extract.py` | Relation extraction |
| 4 | `stages/stage4_filter_wikidata.py` | **One-time** Wikidata filter |
| 5 | `stages/stage5_embed_wikidata.py` | **One-time** bge-small-en embedding |
| 6 | `stages/stage6_match_validate.py` | Top-1 NN + LLM yes/no |
| 7+8 | `stages/stage7_8_ontology.py` | Ontology creation + Turtle formatting |
| 9+10 | `stages/stage9_10_kg.py` | KG construction + rdflib parse |
