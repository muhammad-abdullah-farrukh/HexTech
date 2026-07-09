import json
from pathlib import Path

from sqlalchemy import func, select

from resume_parser.db.ingest import make_ingest_fn
from resume_parser.db.models import Education, Project, Resume, Skill, WorkHistory

STRUCTURED = {
    "candidate_name": "Jane Doe",
    "email": "jane@example.com",
    "skills": ["Python", "SQL"],
    "work_history": [
        {
            "company": "Acme",
            "title": "Engineer",
            "start_date": "2020",
            "end_date": "2022",
        },
    ],
    "education": [
        {"institution": "MIT", "degree": "BSc", "graduation_year": "2019"},
    ],
    "projects": [
        {
            "name": "Widget",
            "description": "A widget",
            "technologies": ["Python", "FastAPI"],
        },
    ],
}


def _write_pdf(tmp_path, content: bytes = b"%PDF-1.4 fake resume bytes") -> str:
    path = tmp_path / "resume.pdf"
    path.write_bytes(content)
    return str(path)


def _spec_file(tmp_path, spec: list[dict]) -> str:
    path = tmp_path / "field_spec.json"
    path.write_text(json.dumps(spec))
    return str(path)


def test_fresh_ingest_creates_rows(session_factory, tmp_path):
    ingest = make_ingest_fn(session_factory, _spec_file(tmp_path, [{"name": "email"}]))
    resume_id = ingest(STRUCTURED, _write_pdf(tmp_path))

    with session_factory() as session:
        resume = session.execute(select(Resume)).scalar_one()
        assert str(resume.id) == resume_id
        assert resume.structured["email"] == "jane@example.com"

        skills = {s.skill for s in session.execute(select(Skill)).scalars()}
        assert skills == {"Python", "SQL"}

        work = session.execute(select(WorkHistory)).scalars().all()
        assert len(work) == 1
        assert work[0].company == "Acme"
        assert work[0].end_date == "2022"

        education = session.execute(select(Education)).scalars().all()
        assert len(education) == 1
        assert education[0].institution == "MIT"

        projects = session.execute(select(Project)).scalars().all()
        assert len(projects) == 1
        assert projects[0].name == "Widget"
        assert projects[0].technologies == ["Python", "FastAPI"]


def test_reingest_same_pdf_updates_in_place(session_factory, tmp_path):
    ingest = make_ingest_fn(session_factory, _spec_file(tmp_path, [{"name": "email"}]))
    pdf = _write_pdf(tmp_path)
    first_id = ingest(STRUCTURED, pdf)

    updated = dict(
        STRUCTURED,
        skills=["Rust"],
        work_history=[{"company": "NewCo", "title": "Lead"}],
    )
    second_id = ingest(updated, pdf)

    assert first_id == second_id
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Resume)) == 1
        # Projection reflects the new data, not a union with the old.
        skills = {s.skill for s in session.execute(select(Skill)).scalars()}
        assert skills == {"Rust"}
        companies = [w.company for w in session.execute(select(WorkHistory)).scalars()]
        assert companies == ["NewCo"]


def test_field_spec_hash_recorded_and_changes(session_factory, tmp_path):
    spec = _spec_file(tmp_path, [{"name": "email"}])
    ingest = make_ingest_fn(session_factory, spec)
    pdf = _write_pdf(tmp_path)

    ingest(STRUCTURED, pdf)
    with session_factory() as session:
        first_hash = session.execute(select(Resume.field_spec_hash)).scalar_one()
    assert first_hash

    Path(spec).write_text(json.dumps([{"name": "email"}, {"name": "phone"}]))
    ingest(STRUCTURED, pdf)
    with session_factory() as session:
        second_hash = session.execute(select(Resume.field_spec_hash)).scalar_one()
    assert second_hash != first_hash


def test_missing_optional_fields_do_not_raise(session_factory, tmp_path):
    ingest = make_ingest_fn(session_factory, _spec_file(tmp_path, [{"name": "email"}]))
    minimal = {"candidate_name": "No Extras", "email": "x@y.com", "skills": ["Go"]}
    ingest(minimal, _write_pdf(tmp_path))

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(WorkHistory)) == 0
        assert session.scalar(select(func.count()).select_from(Education)) == 0
        assert session.scalar(select(func.count()).select_from(Project)) == 0
        skills = {s.skill for s in session.execute(select(Skill)).scalars()}
        assert skills == {"Go"}
