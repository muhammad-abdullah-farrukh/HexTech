"""Turn a stored résumé's `structured` dict into pipeline inputs.

Two functions, one per extraction path:

  render_resume_text()     — Path B input: the résumé's *free text* only
                             (project descriptions, and defensively a summary /
                             work-history descriptions if a future field spec
                             adds them). Everything already broken into named
                             fields is deliberately excluded — those go to Path A.

  structured_to_relations() — Path A: every already-structured field turned into
                             a relation dict, deterministically, with no LLM.
                             Each dict carries both the ontology-shaping fields
                             (property/description, merged before Stage 6) and the
                             concrete triple (subject/object/object_type) that
                             db.kg_staging.stage_structured_relations() writes
                             straight into the graph tables.

The confirmed field spec has free text only in projects[].description, so that's
what render_resume_text() emits today; the summary / work_history description
handling is defensive for forward-compatibility.
"""
from __future__ import annotations


def render_resume_text(structured: dict) -> str:
    """Free-text fields only, for CQ-driven (Path B) extraction."""
    parts: list[str] = []

    summary = (structured.get("summary") or "").strip()
    if summary:
        parts.append(summary)

    for job in structured.get("work_history") or []:
        desc = (job.get("description") or "").strip()
        if desc:
            parts.append(desc)

    for proj in structured.get("projects") or []:
        desc = (proj.get("description") or "").strip()
        if desc:
            name = (proj.get("name") or "").strip()
            parts.append(f"{name}: {desc}" if name else desc)

    return "\n\n".join(parts)


def _rel(property: str, description: str, subject: str, obj: str, object_type: str,
         subject_type: str = "person", object_entity_type: str | None = None) -> dict:
    return {
        "property": property,
        "description": description,
        "subject": subject,
        "subject_type": subject_type,
        "object": obj,
        "object_type": object_type,
        "object_entity_type": object_entity_type,
        "source": "structured",
    }


def _years_experience_text(ye: dict | None) -> str | None:
    if not isinstance(ye, dict):
        return None
    years = ye.get("years")
    months = ye.get("months")
    bits = []
    if years:
        bits.append(f"{years} year{'s' if years != 1 else ''}")
    if months:
        bits.append(f"{months} month{'s' if months != 1 else ''}")
    return " ".join(bits) or None


def structured_to_relations(structured: dict) -> list[dict]:
    """Deterministic relation extraction (Path A) — no LLM call.

    Covers *every* structured field: the Person node's own literals, work
    history, education, skills, and projects (incl. per-technology edges hung off
    the project node). projects[].description is intentionally left to Path B.
    """
    owner = (structured.get("candidate_name") or "").strip() or "Unknown"
    rels: list[dict] = []

    # ── Person node literals (make the owner an actual node with properties) ──
    email = (structured.get("email") or "").strip()
    if email:
        rels.append(_rel("email", "The email address of the person.", owner, email, "literal"))
    phone = (structured.get("phone") or "").strip()
    if phone:
        rels.append(_rel("phone", "The contact phone number of the person.", owner, phone, "literal"))
    ye_text = _years_experience_text(structured.get("years_experience"))
    if ye_text:
        rels.append(_rel(
            "yearsExperience",
            "Total professional experience of the person.",
            owner, ye_text, "literal",
        ))

    # ── Work history ──────────────────────────────────────────────────────────
    for job in structured.get("work_history") or []:
        company = (job.get("company") or "").strip()
        if company:
            rels.append(_rel(
                "employer", "The organization the person worked at.",
                owner, company, "entity", object_entity_type="company",
            ))
        title = (job.get("title") or "").strip()
        if title:
            rels.append(_rel("jobTitle", "A job title the person held.", owner, title, "literal"))
        start = (job.get("start_date") or "").strip()
        if start:
            rels.append(_rel("startDate", "The start date of a role.", owner, start, "literal"))
        end = (job.get("end_date") or "").strip()
        if end:
            rels.append(_rel("endDate", "The end date of a role.", owner, end, "literal"))

    # ── Education ─────────────────────────────────────────────────────────────
    for edu in structured.get("education") or []:
        institution = (edu.get("institution") or "").strip()
        if institution:
            rels.append(_rel(
                "educatedAt", "An institution the person was educated at.",
                owner, institution, "entity", object_entity_type="university",
            ))
        degree = (edu.get("degree") or "").strip()
        if degree:
            rels.append(_rel("degree", "A degree the person earned.", owner, degree, "literal"))
        grad = (edu.get("graduation_year") or "").strip()
        if grad:
            rels.append(_rel("graduationYear", "The graduation year of a degree.", owner, grad, "literal"))

    # ── Skills ────────────────────────────────────────────────────────────────
    for skill in structured.get("skills") or []:
        skill = (skill or "").strip()
        if skill:
            rels.append(_rel(
                "hasSkill", "A skill the person has.",
                owner, skill, "entity", object_entity_type="skill",
            ))

    # ── Projects (name → owner; technologies → the project node) ─────────────
    for proj in structured.get("projects") or []:
        name = (proj.get("name") or "").strip()
        if not name:
            continue
        rels.append(_rel(
            "hasProject", "A project the person built or contributed to.",
            owner, name, "entity", object_entity_type="project",
        ))
        for tech in proj.get("technologies") or []:
            tech = (tech or "").strip()
            if tech:
                rels.append(_rel(
                    "usesTechnology", "A technology used in a project.",
                    name, tech, "entity",
                    subject_type="project", object_entity_type="technology",
                ))

    return rels
