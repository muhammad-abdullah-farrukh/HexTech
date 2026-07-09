"""
pipeline.py — run all stages end-to-end for one or all résumés, reading from and
writing to Postgres (the same database ocr_resume_parser fills).

Usage:
    python pipeline.py                 # all résumés in the resumes table
    python pipeline.py <resume_uuid>   # a single résumé by id
    python pipeline.py --resume        # skip résumés already fully processed
    python pipeline.py <uuid> -r       # single résumé, resume mode
    python pipeline.py <uuid> --from-stage=<stage>
                                        # single résumé; stages before <stage>
                                        # are loaded from outputs/stages/<uuid>/
                                        # instead of recomputed. <stage> is one
                                        # of: cq_gen, cq_answer, relation_extract,
                                        # match_validate, edc_canon, ontology,
                                        # stage9_10 (must be = , not a separate
                                        # token, and requires a uuid — no batch
                                        # form).

Two extraction paths converge in the graph_entities/graph_relationships tables:
    Path A (deterministic, no LLM):  structured_to_relations() → concrete triples
                                     written straight via db.kg_staging.
    Path B (CQ-driven, LLM):         Stages 1→9/10 over the résumé's free text
                                     (render_resume_text), staged after Stage 9.
Both paths' relation *dicts* also merge before Stage 6 to shape the ontology.

Stage order (Path B):
    1   CQ generation            2   CQ answering        3   Relation extraction
    3.5 EDC canonicalization     6   Wikidata match      7/8 Ontology
    9/10 KG construction → staged into Postgres

Resume mode (-r / --resume): each stage's cached output is read from the
pipeline_runs table instead of re-running; a résumé with a succeeded 'kg_facts'
row is skipped entirely.
"""
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone

from config import DATABASE_URL, LLM_BASE_URL, OUTPUTS_DIR, LLM_ENDPOINTS
from db.session import make_session_factory
from db import runs, kg_staging
from render import render_resume_text, structured_to_relations
from resume_parser.db.models import Resume
from stages import (
    canonicalize,
    stage1_cq_gen,
    stage2_cq_answer,
    stage3_relation_extract,
    stage6_match_validate,
    stage7_8_ontology,
    stage9_10_kg,
)


def _log(resume_id, msg: str) -> None:
    """Append a timestamped line to a plain-text log outside Postgres, so the
    last stage that ran for a résumé can be checked (`tail`/`cat`) without a
    DB connection — a companion to, not a replacement for, pipeline_runs."""
    log_path = OUTPUTS_DIR / "logs" / f"pipeline_{resume_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(log_path, "a") as f:
        f.write(f"{ts} {msg}\n")


# Numbered on-disk filenames mirroring the stage order documented at the top
# of this module, so a résumé's outputs/stages/<uuid>/ directory can be read
# stage-by-stage without a DB connection.
_STAGE_FILENAMES = {
    "cq_gen": "stage1_cq_gen.json",
    "cq_answer": "stage2_cq_answer.json",
    "relation_extract": "stage3_relation_extract.json",
    "edc_canon": "stage3_5_edc_canon.json",
    "match_validate": "stage6_match_validate.json",
    "ontology": "stage7_8_ontology.json",
    "kg_facts": "stage9_10_kg_facts_marker.json",
}

# The post-EDC match_results artifact (see _run_edc_canonicalization) — not a
# pipeline_runs stage key, just a disk-only filename used by --from-stage.
_EDC_RESULT_FILENAME = "stage3_5_edc_canon_result.json"

# Actual runtime dependency order (match_validate runs before edc_canon in
# process_resume, despite the module docstring's stage numbering) — drives
# --from-stage: every stage strictly before the chosen one is loaded from
# disk instead of recomputed.
_STAGE_ORDER = [
    "cq_gen", "cq_answer", "relation_extract",
    "match_validate", "edc_canon", "ontology", "stage9_10",
]


def _save_stage_to_disk(resume_id, filename: str, data) -> None:
    """Mirror a stage's output to outputs/stages/<resume_id>/<filename>, purely
    as a convenience artifact alongside pipeline_runs / graph_entities /
    graph_relationships — nothing in this pipeline reads it back, so it's safe
    to inspect or delete without affecting resume mode."""
    stage_dir = OUTPUTS_DIR / "stages" / str(resume_id)
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / filename
    if filename.endswith(".ttl"):
        path.write_text(data if isinstance(data, str) else str(data))
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def _load_stage_from_disk(resume_id, filename: str):
    """Inverse of _save_stage_to_disk, used by --from-stage to skip
    recomputing earlier stages."""
    path = OUTPUTS_DIR / "stages" / str(resume_id) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"no on-disk output at {path} — run without --from-stage, or "
            f"from an earlier stage, first"
        )
    if filename.endswith(".ttl"):
        return path.read_text()
    with open(path) as f:
        return json.load(f)


def _cached_or_run(session_factory, resume_id, stage, resume, fn):
    """Return this stage's cached pipeline_runs output in resume mode, else run
    `fn()`, persist its output, and return it."""
    if resume:
        with session_factory() as session:
            cached = runs.get_stage_output(session, resume_id, stage)
        if cached is not None:
            print(f"[pipeline] resumed '{stage}' from pipeline_runs")
            _log(resume_id, f"STAGE {stage}: resumed from pipeline_runs cache")
            if stage in _STAGE_FILENAMES:
                _save_stage_to_disk(resume_id, _STAGE_FILENAMES[stage], cached)
            return cached
    _log(resume_id, f"STAGE {stage}: started")
    result = fn()
    with session_factory() as session:
        runs.save_stage_output(session, resume_id, stage, result)
    if stage in _STAGE_FILENAMES:
        _save_stage_to_disk(resume_id, _STAGE_FILENAMES[stage], result)
    _log(resume_id, f"STAGE {stage}: succeeded")
    return result


def _run_edc_canonicalization(session_factory, match_results, cqs, resume_id, resume):
    """
    EDC step (3.5): for unmatched relations (wikidata_match is None), check the
    EDC canon store (Postgres) for a cross-document merge candidate. Annotates
    each match_result with "canon_match" and "edc_definition".

    Deduplicated by property label before calling EDC at all — Path A
    (structured_to_relations) emits one relation instance per skill/project/
    work-history entry, so a real résumé can carry the SAME label dozens of
    times (e.g. 'hasSkill' once per skill listed) even though the property
    means exactly the same thing every occurrence. Stage 6
    (stage6_match_validate.py) already does this same dedup-by-property-name
    for its own validation step; EDC just never had the equivalent. One
    representative occurrence per unique label is canonicalized, and the
    result is broadcast to every occurrence sharing that label — cuts real
    EDC volume by ~4-5x on a typical résumé with no change in correctness
    (the label's meaning doesn't depend on which specific value it's
    attached to).

    Checkpointed per-UNIQUE-LABEL (not per relation instance) under stage
    'edc_canon' in pipeline_runs — each label's result is persisted
    immediately after it completes, not just once at the end, so a
    killed/restarted run loses at most the one label that was in flight.
    The checkpoint is only read back (to skip re-calling the LLM) in resume
    mode (-r); a fresh run still writes it as it goes, but always
    reprocesses — the canon store may have grown since, enabling new merges
    a stale checkpoint wouldn't know about.
    """
    try:
        from stages.canonicalize import canonicalize_relation
        from db import canon as canon_db

        unmatched = [item for item in match_results if item["wikidata_match"] is None]
        if not unmatched:
            return match_results

        unmatched_by_label: dict[str, list[dict]] = {}
        for item in match_results:
            if item["wikidata_match"] is not None:
                item.setdefault("canon_match", None)
                item.setdefault("edc_definition", "")
                continue
            unmatched_by_label.setdefault(item["extracted"]["property"], []).append(item)

        print(
            f"[EDC]  {len(unmatched)} unmatched relation instance(s), "
            f"{len(unmatched_by_label)} unique propert(ies) after dedup "
            f"→ EDC canon store lookup", flush=True,
        )
        _log(resume_id, f"STAGE edc_canon: started ({len(unmatched)} instances, "
                         f"{len(unmatched_by_label)} unique properties)")

        checkpoint: dict = {}
        if resume:
            with session_factory() as session:
                cached = runs.get_stage_output(session, resume_id, runs.EDC_CANON)
            if cached:
                checkpoint = cached
                print(f"[EDC]  resuming — {len(checkpoint)} propert(ies) already checkpointed", flush=True)
                _log(resume_id, f"STAGE edc_canon: resumed {len(checkpoint)} checkpointed propert(ies)")

        # Each unique label is canonicalized independently (the canon store is
        # only READ here — new properties are registered later, in Stage 7/8 —
        # so no label observes another's result within this stage). Fan the
        # LLM-heavy per-label work out across endpoints; already-checkpointed
        # labels (resume mode) are reused without a call.
        from concurrent.futures import ThreadPoolExecutor
        import threading

        to_compute = [
            (label, items) for label, items in unmatched_by_label.items()
            if checkpoint.get(label) is None
        ]
        result_by_label: dict[str, dict] = {
            label: checkpoint[label]
            for label in unmatched_by_label
            if checkpoint.get(label) is not None
        }
        ckpt_lock = threading.Lock()

        def _canon_one(label: str, items: list[dict], endpoint: dict) -> tuple[str, dict]:
            result = canonicalize_relation(items[0]["extracted"], cqs, str(resume_id), endpoint=endpoint)
            result_dict = asdict(result)
            if result.was_merged and result.canonical_label is not None:
                print(
                    f"[EDC]   '{label}' → merged with "
                    f"'{result.canonical_label}' (conf={result.confidence:.2f}) "
                    f"[{len(items)} occurrence(s)]",
                    flush=True,
                )
            # Persist immediately (per label) — bounds data loss on a kill to
            # whatever labels are still in flight, not everything still to come.
            with ckpt_lock:
                checkpoint[label] = result_dict
                with session_factory() as session:
                    runs.save_stage_output(session, resume_id, runs.EDC_CANON, checkpoint)
                _save_stage_to_disk(resume_id, _STAGE_FILENAMES[runs.EDC_CANON], checkpoint)
            return label, result_dict

        if to_compute:
            total_slots = sum(max(1, int(e.get("slots", 1))) for e in LLM_ENDPOINTS)
            with ThreadPoolExecutor(max_workers=max(1, min(total_slots, len(to_compute)))) as pool:
                futures = [
                    pool.submit(_canon_one, label, items, LLM_ENDPOINTS[i % len(LLM_ENDPOINTS)])
                    for i, (label, items) in enumerate(to_compute)
                ]
                for fut in futures:
                    label, result_dict = fut.result()
                    result_by_label[label] = result_dict

        # Broadcast each label's result to every occurrence of that label.
        for label, items in unmatched_by_label.items():
            result_dict = result_by_label[label]
            canon_match = None
            if result_dict["was_merged"] and result_dict["canonical_label"]:
                with session_factory() as session:
                    canon_match = canon_db.find_by_label(session, result_dict["canonical_label"])
            for item in items:
                item["edc_definition"] = result_dict["definition"]
                item["canon_match"] = canon_match

        _log(resume_id, f"STAGE edc_canon: succeeded ({len(checkpoint)} unique propert(ies) processed)")

    except Exception as e:
        print(f"[EDC]  ⚠ EDC canonicalization skipped: {e}", flush=True)
        _log(resume_id, f"STAGE edc_canon: FAILED — {e}")
        for item in match_results:
            item.setdefault("canon_match", None)
            item.setdefault("edc_definition", "")

    return match_results


def _skip_before(from_stage: str | None, stage: str) -> bool:
    """True if `stage` should be skipped (its output loaded from disk instead
    of recomputed) because --from-stage asked to start at a later stage."""
    if from_stage is None:
        return False
    return _STAGE_ORDER.index(stage) < _STAGE_ORDER.index(from_stage)


def _stage_value(session_factory, resume_id, stage, resume, from_stage, fn):
    """Return a stage's output: loaded from its outputs/stages/ disk file if
    --from-stage skips it, else via the normal pipeline_runs cache-or-run
    path (_cached_or_run). Backfills pipeline_runs on the disk-load path too,
    so -r and disk stay consistent afterward."""
    if _skip_before(from_stage, stage):
        value = _load_stage_from_disk(resume_id, _STAGE_FILENAMES[stage])
        with session_factory() as session:
            runs.save_stage_output(session, resume_id, stage, value)
        _log(resume_id, f"STAGE {stage}: loaded from disk (--from-stage)")
        return value
    return _cached_or_run(session_factory, resume_id, stage, resume, fn)


def process_resume(resume_id, session_factory, resume: bool = False, from_stage: str | None = None):
    doc_name = str(resume_id)
    with session_factory() as session:
        row = session.get(Resume, resume_id)
        if row is None:
            print(f"[pipeline] ⚠ résumé {resume_id} not found — skipping.")
            return
        structured = row.structured

    print(f"\n{'='*60}\nProcessing résumé: {doc_name}\n{'='*60}")
    _log(resume_id, "PIPELINE: run started")

    narrative_text = render_resume_text(structured)
    struct_rels = structured_to_relations(structured)
    print(f"[pipeline] Path A: {len(struct_rels)} structured relation(s); "
          f"Path B free-text: {len(narrative_text)} chars")

    # ── Path B: CQ-driven stages over free text only ──────────────────────
    cqs, qa_pairs, cq_rels = [], [], []
    if narrative_text.strip():
        cqs = _stage_value(
            session_factory, resume_id, runs.CQ_GEN, resume, from_stage,
            lambda: stage1_cq_gen.run(doc_name, narrative_text),
        )
        if cqs:
            qa_pairs = _stage_value(
                session_factory, resume_id, runs.CQ_ANSWER, resume, from_stage,
                lambda: stage2_cq_answer.run(doc_name, narrative_text, cqs),
            )
            cq_rels = _stage_value(
                session_factory, resume_id, runs.RELATION_EXTRACT, resume, from_stage,
                lambda: stage3_relation_extract.run(doc_name, narrative_text, cqs),
            )
    else:
        print("[pipeline] No free text to mine — Path B skipped, Path A only.")

    # ── Path B relations only feed Stage 6/EDC/ontology ────────────────────
    # struct_rels (Path A) used to be merged in here too, but Path A's own
    # graph write (stage_structured_relations, below) never reads
    # match_results/ontology at all — its relationship types come from a
    # static regex (_rel_type) and its entity values are resolved separately
    # via ResumeEntityResolver. Path A's property set is also fixed by the
    # parser's schema (email, employer, hasSkill, ...), not per-résumé
    # content, so re-validating it against Wikidata/EDC every run was pure
    # waste — for a résumé with many skills/projects, struct_rels could emit
    # the same property dozens of times, each a redundant pass through this
    # pipeline (see the hasSkill retry-storm this caused with 41 skills).
    relations = list(cq_rels)

    ontology = None
    if relations:
        # --from-stage=ontology or later: match_validate + edc_canon both
        # already ran: load the fully-augmented match_results directly
        # instead of recomputing either.
        need_match_results = not _skip_before(from_stage, "ontology")
        if need_match_results:
            if _skip_before(from_stage, "edc_canon"):
                match_results = _load_stage_from_disk(resume_id, _EDC_RESULT_FILENAME)
                _log(resume_id, "STAGE match_validate+edc_canon: loaded post-EDC "
                                 "match_results from disk (--from-stage)")
            else:
                def _match():
                    with session_factory() as session:
                        return stage6_match_validate.run(session, relations, doc_name=doc_name)

                match_results = _stage_value(
                    session_factory, resume_id, runs.MATCH_VALIDATE, resume, from_stage, _match
                )
                match_results = _run_edc_canonicalization(
                    session_factory, match_results, cqs, resume_id, resume
                )
                _save_stage_to_disk(resume_id, _EDC_RESULT_FILENAME, match_results)

        if _skip_before(from_stage, "ontology"):
            ontology = _load_stage_from_disk(resume_id, "stage7_8_ontology.ttl")
            with session_factory() as session:
                runs.save_stage_output(session, resume_id, runs.ONTOLOGY, {"ttl": ontology})
            _log(resume_id, "STAGE ontology: loaded from disk (--from-stage)")
        else:
            ontology = _cached_or_run(
                session_factory, resume_id, runs.ONTOLOGY, resume,
                lambda: {"ttl": stage7_8_ontology.run(resume_id, doc_name, match_results)},
            )
            ontology = ontology.get("ttl") if isinstance(ontology, dict) else ontology
        if ontology:
            _save_stage_to_disk(resume_id, "stage7_8_ontology.ttl", ontology)

    # ── Path B KG (Stage 9/10) — staged into Postgres ────────────────────
    if ontology and qa_pairs:
        with session_factory() as session:
            g, triples = stage9_10_kg.run(
                session, resume_id, doc_name, narrative_text, qa_pairs, ontology
            )
        if g is not None:
            _save_stage_to_disk(
                resume_id, "stage9_10_kg_facts.ttl", g.serialize(format="turtle")
            )
            _save_stage_to_disk(
                resume_id, "stage9_10_kg_facts.json", {"n_triples": len(triples)}
            )

    # ── Path A: concrete structured triples straight into the staging tables
    resolver = canonicalize.get_entity_resolver()
    with session_factory() as session:
        kg_staging.stage_structured_relations(session, resume_id, struct_rels, resolver=resolver)
    print(f"[pipeline] Path A: {len(struct_rels)} structured relation(s) staged")
    _log(resume_id, f"STAGE path_a: succeeded ({len(struct_rels)} structured relation(s) staged)")
    _save_stage_to_disk(resume_id, "path_a_structured_relations.json", struct_rels)

    with session_factory() as session:
        runs.save_stage_output(session, resume_id, runs.KG_FACTS, {"staged": True})
    _save_stage_to_disk(resume_id, _STAGE_FILENAMES[runs.KG_FACTS], {"staged": True})
    print(f"[pipeline] Done: {doc_name}")
    _log(resume_id, "PIPELINE: run finished (kg_facts marked staged)")


def _resume_ids(session_factory, resume: bool) -> list:
    """All résumé ids, optionally skipping those already fully processed."""
    from sqlalchemy import select
    from db.models import PipelineRun

    with session_factory() as session:
        stmt = select(Resume.id)
        if resume:
            done = select(PipelineRun.document_id).where(
                PipelineRun.stage == runs.KG_FACTS, PipelineRun.status == "succeeded"
            )
            stmt = stmt.where(Resume.id.not_in(done))
        return list(session.execute(stmt).scalars())


def main():
    from_stage = None
    rest = []
    for a in sys.argv[1:]:
        if a.startswith("--from-stage="):
            from_stage = a.split("=", 1)[1]
        else:
            rest.append(a)

    flags = {a for a in rest if a.startswith("-")}
    posargs = [a for a in rest if not a.startswith("-")]
    resume = "--resume" in flags or "-r" in flags

    if from_stage is not None and from_stage not in _STAGE_ORDER:
        print(f"[pipeline] ✗ unknown --from-stage={from_stage!r}; "
              f"expected one of {_STAGE_ORDER}")
        return
    if from_stage is not None and not posargs:
        print("[pipeline] ✗ --from-stage requires a résumé uuid "
              "(it doesn't apply across a whole batch).")
        return

    if not DATABASE_URL:
        print("[pipeline] ✗ DATABASE_URL is not set (export it or add to .env).")
        return

    session_factory = make_session_factory(DATABASE_URL)
    canonicalize.configure(session_factory)

    # Guard the single --parallel 1 LLM slot shared with the parser.
    from resume_parser.llm_lock import llm_lock

    with llm_lock(LLM_BASE_URL):
        if posargs:
            ids = [posargs[0]]
        else:
            ids = _resume_ids(session_factory, resume)
            if not ids:
                print("[pipeline] No résumés to process.")
                return

        if resume:
            print("[pipeline] Resume mode: skipping stages cached in pipeline_runs.")
        if from_stage:
            print(f"[pipeline] --from-stage={from_stage}: earlier stages loaded from disk.")

        for rid in ids:
            process_resume(rid, session_factory, resume=resume, from_stage=from_stage)


if __name__ == "__main__":
    main()
