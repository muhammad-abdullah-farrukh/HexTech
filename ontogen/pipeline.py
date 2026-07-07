"""
pipeline.py — run all stages end-to-end for one or all résumés, reading from and
writing to Postgres (the same database ocr_resume_parser fills).

Usage:
    python pipeline.py                 # all résumés in the resumes table
    python pipeline.py <resume_uuid>   # a single résumé by id
    python pipeline.py --resume        # skip résumés already fully processed
    python pipeline.py <uuid> -r       # single résumé, resume mode

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
import sys

from config import DATABASE_URL, LLM_BASE_URL
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


def _cached_or_run(session_factory, resume_id, stage, resume, fn):
    """Return this stage's cached pipeline_runs output in resume mode, else run
    `fn()`, persist its output, and return it."""
    if resume:
        with session_factory() as session:
            cached = runs.get_stage_output(session, resume_id, stage)
        if cached is not None:
            print(f"[pipeline] resumed '{stage}' from pipeline_runs")
            return cached
    result = fn()
    with session_factory() as session:
        runs.save_stage_output(session, resume_id, stage, result)
    return result


def _run_edc_canonicalization(session_factory, match_results, cqs):
    """
    EDC step (3.5): for unmatched relations (wikidata_match is None), check the
    EDC canon store (Postgres) for a cross-document merge candidate. Annotates
    each match_result with "canon_match" and "edc_definition". Always re-runs:
    the canon store may have grown since a cached match, enabling new merges.
    """
    try:
        from stages.canonicalize import canonicalize_relation
        from db import canon as canon_db

        unmatched = [item for item in match_results if item["wikidata_match"] is None]
        if not unmatched:
            return match_results

        # Canonicalize each UNIQUE property name once and broadcast the result to
        # every occurrence — mirrors Stage 6's dedup. Without this, a résumé with
        # 34 skills ran 34 identical hasSkill define+verify cycles (each ~1 define
        # + up to k verify LLM calls).
        uniq_names = {item["extracted"]["property"].strip().lower() for item in unmatched}
        print(f"[EDC]  {len(unmatched)} unmatched relation(s), {len(uniq_names)} unique "
              f"property name(s) → EDC canon store lookup", flush=True)

        cache: dict[str, tuple] = {}  # prop_key -> (canon_match, edc_definition)
        for item in match_results:
            if item["wikidata_match"] is not None:
                item.setdefault("canon_match", None)
                item.setdefault("edc_definition", "")
                continue

            key = item["extracted"]["property"].strip().lower()
            if key not in cache:
                result = canonicalize_relation(item["extracted"], cqs)
                canon_match = None
                if result.was_merged and result.canonical_label is not None:
                    with session_factory() as session:
                        canon_match = canon_db.find_by_label(session, result.canonical_label)
                    print(
                        f"[EDC]   '{item['extracted']['property']}' → merged with "
                        f"'{result.canonical_label}' (conf={result.confidence:.2f})",
                        flush=True,
                    )
                cache[key] = (canon_match, result.definition)
            item["canon_match"], item["edc_definition"] = cache[key]

    except Exception as e:
        print(f"[EDC]  ⚠ EDC canonicalization skipped: {e}", flush=True)
        for item in match_results:
            item.setdefault("canon_match", None)
            item.setdefault("edc_definition", "")

    return match_results


def process_resume(resume_id, session_factory, resume: bool = False):
    doc_name = str(resume_id)
    with session_factory() as session:
        row = session.get(Resume, resume_id)
        if row is None:
            print(f"[pipeline] ⚠ résumé {resume_id} not found — skipping.")
            return
        structured = row.structured

    print(f"\n{'='*60}\nProcessing résumé: {doc_name}\n{'='*60}")

    narrative_text = render_resume_text(structured)
    struct_rels = structured_to_relations(structured)
    print(f"[pipeline] Path A: {len(struct_rels)} structured relation(s); "
          f"Path B free-text: {len(narrative_text)} chars")

    # ── Path B: CQ-driven stages over free text only ──────────────────────
    cqs, qa_pairs, cq_rels = [], [], []
    if narrative_text.strip():
        cqs = _cached_or_run(
            session_factory, resume_id, runs.CQ_GEN, resume,
            lambda: stage1_cq_gen.run(doc_name, narrative_text),
        )
        if cqs:
            qa_pairs = _cached_or_run(
                session_factory, resume_id, runs.CQ_ANSWER, resume,
                lambda: stage2_cq_answer.run(doc_name, narrative_text, cqs),
            )
            cq_rels = _cached_or_run(
                session_factory, resume_id, runs.RELATION_EXTRACT, resume,
                lambda: stage3_relation_extract.run(doc_name, narrative_text, cqs),
            )
    else:
        print("[pipeline] No free text to mine — Path B skipped, Path A only.")

    # ── Merge both paths' relation dicts → shape the ontology (Stage 6) ───
    relations = list(cq_rels) + list(struct_rels)

    ontology = None
    if relations:
        def _match():
            with session_factory() as session:
                return stage6_match_validate.run(session, relations, doc_name=doc_name)

        match_results = _cached_or_run(
            session_factory, resume_id, runs.MATCH_VALIDATE, resume, _match
        )
        match_results = _run_edc_canonicalization(session_factory, match_results, cqs)

        ontology = _cached_or_run(
            session_factory, resume_id, runs.ONTOLOGY, resume,
            lambda: {"ttl": stage7_8_ontology.run(resume_id, doc_name, match_results)},
        )
        ontology = ontology.get("ttl") if isinstance(ontology, dict) else ontology

    # ── Path B KG (Stage 9/10) — staged into Postgres ────────────────────
    if ontology and qa_pairs:
        with session_factory() as session:
            stage9_10_kg.run(session, resume_id, doc_name, narrative_text, qa_pairs, ontology)

    # ── Path A: concrete structured triples straight into the staging tables
    resolver = canonicalize.get_entity_resolver()
    with session_factory() as session:
        kg_staging.stage_structured_relations(session, resume_id, struct_rels, resolver=resolver)
    print(f"[pipeline] Path A: {len(struct_rels)} structured relation(s) staged")

    with session_factory() as session:
        runs.save_stage_output(session, resume_id, runs.KG_FACTS, {"staged": True})
    print(f"[pipeline] Done: {doc_name}")


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
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    posargs = [a for a in sys.argv[1:] if not a.startswith("-")]
    resume = "--resume" in flags or "-r" in flags

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

        for rid in ids:
            process_resume(rid, session_factory, resume=resume)


if __name__ == "__main__":
    main()
