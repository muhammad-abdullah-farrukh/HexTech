"""CLI entry point — run the pipeline end-to-end on one PDF.

    python -m resume_parser.cli RESUME.pdf \\
        --field-spec config/field_spec.json \\
        --output resume_extracted.json \\
        --artifacts-dir artifacts/ \\
        [--model SLUG] [--env .env] [--strict] [--no-llm]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .pipeline import run_pipeline
from .schema_builder import load_field_spec
from .settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="resume_parser",
        description="Extract a resume PDF into schema-conformant JSON.",
    )
    p.add_argument("pdf", help="Path to the resume PDF.")
    p.add_argument(
        "--field-spec",
        default="config/field_spec.json",
        help="Path to the field-spec JSON (default: config/field_spec.json).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Where to write the result JSON. If omitted, prints to stdout only.",
    )
    p.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory for raw/cleaned extraction artifacts (skipped if omitted).",
    )
    p.add_argument("--model", default=None, help="Override the LLM model name.")
    p.add_argument("--env", default=None, help="Path to a .env file to load.")
    p.add_argument(
        "--db-uri",
        default=None,
        help=(
            "Persist the structured result to this Postgres URI "
            "(e.g. postgresql+psycopg2://user:pass@host:5432/db). Omit to stay "
            "file-only; ingestion is never inferred from the environment."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Emit a strict-compatible schema (all-required + additionalProperties:false).",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Run extraction + cleanup + artifacts only; skip the LLM call.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    field_spec = load_field_spec(args.field_spec)

    settings = None
    if not args.no_llm:
        settings = load_settings(args.env)
        if args.model:
            from dataclasses import replace

            settings = replace(settings, model=args.model)

    ingest_fn = None
    if args.db_uri:
        from .db.ingest import make_ingest_fn
        from .db.session import make_session_factory

        session_factory = make_session_factory(args.db_uri)
        ingest_fn = make_ingest_fn(session_factory, args.field_spec)
        print(f"Ingesting to database at {args.db_uri}", file=sys.stderr)

    # Guard the single --parallel 1 LLM slot shared with Ontogen. Only needed
    # when we actually call the LLM; --no-llm runs skip the lock entirely.
    if args.no_llm:
        from contextlib import nullcontext

        lock_ctx = nullcontext()
    else:
        from .llm_lock import llm_lock

        lock_ctx = llm_lock(settings.base_url)

    with lock_ctx:
        result = run_pipeline(
            args.pdf,
            field_spec,
            output_path=args.output,
            settings=settings,
            artifacts_dir=args.artifacts_dir,
            run_llm=not args.no_llm,
            strict=args.strict,
            ingest_fn=ingest_fn,
        )

    if args.no_llm:
        print(
            f"Extraction complete (--no-llm). "
            f"Artifacts: {args.artifacts_dir or '(none — pass --artifacts-dir)'}",
            file=sys.stderr,
        )
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
