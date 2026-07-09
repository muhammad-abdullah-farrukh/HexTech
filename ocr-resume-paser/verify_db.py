"""Show how the resume database is built and verify every stored row.

Two parts, written to a Markdown report (default: DB_REPORT.md) and summarized on
stdout:

  1. SCHEMA — introspected live from Postgres: every table's columns, primary key,
     unique constraints, foreign keys, and indexes. This is the actual shape of the
     database, not what the models claim.
  2. VERIFICATION — for each `resumes` row, checks that the derived projection tables
     (skills / work_history / education) exactly match the arrays inside the stored
     `structured` JSONB. The projection is a cache of the JSON, so any mismatch means
     ingestion is wrong. Each resume gets a PASS/FAIL with per-field detail.

Usage:
    python verify_db.py [--env .env] [--db-uri URI] [--output DB_REPORT.md]

Reads DATABASE_URL from the environment / .env when --db-uri is omitted.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, func, inspect, select

from resume_parser.db.models import Education, Project, Resume, Skill, WorkHistory

TABLES = ["resumes", "skills", "work_history", "education", "projects"]


def _schema_section(engine) -> list[str]:
    """Render the live schema (columns, keys, indexes) as Markdown lines."""
    inspector = inspect(engine)
    lines = ["## Schema (introspected from Postgres)", ""]
    for table in TABLES:
        if table not in inspector.get_table_names():
            lines.append(f"### `{table}` — MISSING (run `alembic upgrade head`)\n")
            continue
        lines.append(f"### `{table}`\n")
        lines.append("| column | type | nullable | default |")
        lines.append("| --- | --- | --- | --- |")
        for col in inspector.get_columns(table):
            default = col.get("default") or ""
            lines.append(
                f"| {col['name']} | {col['type']} | {col['nullable']} | {default} |"
            )
        pk = inspector.get_pk_constraint(table).get("constrained_columns", [])
        uniques = [
            ",".join(u["column_names"]) for u in inspector.get_unique_constraints(table)
        ]
        fks = [
            f"{','.join(fk['constrained_columns'])} -> "
            f"{fk['referred_table']}({','.join(fk['referred_columns'])}) "
            f"ON DELETE {fk.get('options', {}).get('ondelete', '-')}"
            for fk in inspector.get_foreign_keys(table)
        ]
        indexes = [ix["name"] for ix in inspector.get_indexes(table)]
        lines.append("")
        lines.append(f"- Primary key: `{', '.join(pk) or '-'}`")
        lines.append(f"- Unique: `{'; '.join(uniques) or '-'}`")
        lines.append(f"- Foreign keys: `{'; '.join(fks) or '-'}`")
        lines.append(f"- Indexes: `{', '.join(indexes) or '-'}`")
        lines.append("")
    return lines


def _verify_resume(session, resume: Resume) -> tuple[bool, list[str]]:
    """Compare one resume's projection rows to its stored JSON; return (ok, detail)."""
    structured = resume.structured
    detail: list[str] = []
    ok = True

    json_skills = Counter(structured.get("skills") or [])
    row_skills = Counter(
        s.skill
        for s in session.execute(
            select(Skill).where(Skill.resume_id == resume.id)
        ).scalars()
    )
    skills_ok = json_skills == row_skills
    ok &= skills_ok
    detail.append(
        f"- skills: {'OK' if skills_ok else 'MISMATCH'} "
        f"({sum(row_skills.values())} rows vs {sum(json_skills.values())} in JSON)"
    )

    json_work = Counter(
        (e.get("company"), e.get("title"), e.get("start_date"), e.get("end_date"))
        for e in (structured.get("work_history") or [])
    )
    row_work = Counter(
        (w.company, w.title, w.start_date, w.end_date)
        for w in session.execute(
            select(WorkHistory).where(WorkHistory.resume_id == resume.id)
        ).scalars()
    )
    work_ok = json_work == row_work
    ok &= work_ok
    detail.append(
        f"- work_history: {'OK' if work_ok else 'MISMATCH'} "
        f"({sum(row_work.values())} rows vs {sum(json_work.values())} in JSON)"
    )

    json_edu = Counter(
        (e.get("institution"), e.get("degree"), e.get("graduation_year"))
        for e in (structured.get("education") or [])
    )
    row_edu = Counter(
        (e.institution, e.degree, e.graduation_year)
        for e in session.execute(
            select(Education).where(Education.resume_id == resume.id)
        ).scalars()
    )
    edu_ok = json_edu == row_edu
    ok &= edu_ok
    detail.append(
        f"- education: {'OK' if edu_ok else 'MISMATCH'} "
        f"({sum(row_edu.values())} rows vs {sum(json_edu.values())} in JSON)"
    )

    json_proj = Counter(
        (e.get("name"), e.get("description"), tuple(e.get("technologies") or []))
        for e in (structured.get("projects") or [])
    )
    row_proj = Counter(
        (p.name, p.description, tuple(p.technologies or []))
        for p in session.execute(
            select(Project).where(Project.resume_id == resume.id)
        ).scalars()
    )
    proj_ok = json_proj == row_proj
    ok &= proj_ok
    detail.append(
        f"- projects: {'OK' if proj_ok else 'MISMATCH'} "
        f"({sum(row_proj.values())} rows vs {sum(json_proj.values())} in JSON)"
    )
    return ok, detail


def build_report(database_url: str) -> tuple[str, bool]:
    """Return (markdown_report, all_passed)."""
    engine = create_engine(database_url)
    lines = ["# Resume Database Report", ""]
    lines += _schema_section(engine)

    all_ok = True
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        resumes = (
            session.execute(select(Resume).order_by(Resume.ingested_at)).scalars().all()
        )

        # Per-resume counts of what was actually stored in each projection table.
        summary = ["## What was stored per resume", ""]
        summary.append(f"Stored resumes: **{len(resumes)}**\n")
        summary.append(
            "| Candidate | Email | Skills | Work history | Education | Projects |"
        )
        summary.append("| --- | --- | ---: | ---: | ---: | ---: |")
        totals = {"skills": 0, "work": 0, "edu": 0, "proj": 0}
        detail_lines = ["## Verification (projection tables vs stored JSON)", ""]
        for resume in resumes:
            rid = resume.id
            n_skills = session.scalar(
                select(func.count()).select_from(Skill).where(Skill.resume_id == rid)
            )
            n_work = session.scalar(
                select(func.count())
                .select_from(WorkHistory)
                .where(WorkHistory.resume_id == rid)
            )
            n_edu = session.scalar(
                select(func.count())
                .select_from(Education)
                .where(Education.resume_id == rid)
            )
            n_proj = session.scalar(
                select(func.count()).select_from(Project).where(Project.resume_id == rid)
            )
            totals["skills"] += n_skills
            totals["work"] += n_work
            totals["edu"] += n_edu
            totals["proj"] += n_proj

            name = resume.structured.get("candidate_name", "(no name)")
            email = resume.structured.get("email", "(no email)")
            summary.append(
                f"| {name} | {email} | {n_skills} | {n_work} | {n_edu} | {n_proj} |"
            )

            ok, detail = _verify_resume(session, resume)
            all_ok &= ok
            detail_lines.append(f"### {'✅' if ok else '❌'} {name} — {email}")
            detail_lines.append(f"- source_file: `{resume.source_file}`")
            detail_lines.append(f"- id: `{resume.id}`")
            detail_lines.append(f"- pdf_hash: `{resume.pdf_hash}`")
            detail_lines.append(f"- field_spec_hash: `{resume.field_spec_hash}`")
            detail_lines.append(f"- ingested_at: `{resume.ingested_at}`")
            detail_lines += detail
            detail_lines.append("")

        summary.append(
            f"| **Total** | | **{totals['skills']}** | **{totals['work']}** | "
            f"**{totals['edu']}** | **{totals['proj']}** |"
        )
        summary.append("")
        lines += summary
        lines += detail_lines

    verdict = "ALL RESUMES VERIFIED" if all_ok else "MISMATCHES FOUND — see above"
    lines.append(f"## Result: {verdict}")
    lines.append("")
    engine.dispose()
    return "\n".join(lines), all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_db",
        description="Report the resume DB schema and verify stored rows against their JSON.",
    )
    parser.add_argument("--env", default=None, help="Path to a .env file to load.")
    parser.add_argument(
        "--db-uri", default=None, help="Postgres URI (default: DATABASE_URL from env)."
    )
    parser.add_argument(
        "--output",
        default="docs/DB_REPORT.md",
        help="Report path (default: docs/DB_REPORT.md).",
    )
    args = parser.parse_args(argv)

    load_dotenv(args.env) if args.env else load_dotenv()
    database_url = args.db_uri or os.environ.get("DATABASE_URL")
    if not database_url:
        print(
            "No database URI. Pass --db-uri or set DATABASE_URL (see .env.example).",
            file=sys.stderr,
        )
        return 2

    report, all_ok = build_report(database_url)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nWrote {args.output}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
