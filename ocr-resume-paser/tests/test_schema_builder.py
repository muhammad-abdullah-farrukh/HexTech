import pytest

from resume_parser.schema_builder import (
    build_dynamic_model,
    build_review_model,
    order_by_spec,
    render_field_guide,
    strip_defaults,
)

SPEC = [
    {"name": "candidate_name", "type": "string", "required": True},
    {"name": "phone", "type": "string", "required": False},
    {"name": "years_experience", "type": "integer", "required": True},
    {"name": "skills", "type": "array", "items": "string", "required": True},
    {
        "name": "work_history",
        "type": "array",
        "items": "object",
        "required": False,
        "properties": [
            {"name": "company", "type": "string", "required": True},
            {"name": "title", "type": "string", "required": True},
            {"name": "start_date", "type": "string", "required": False},
        ],
    },
]


def test_required_and_optional_and_nested_types():
    M = build_dynamic_model("Resume", SPEC)
    m = M(
        candidate_name="Jane",
        years_experience=5,
        skills=["python"],
        work_history=[{"company": "Acme", "title": "SWE"}],
    )
    assert m.phone is None  # optional defaults to None
    assert m.skills == ["python"]
    # Object-array compiled to a real nested model, not a raw dict.
    assert m.work_history[0].company == "Acme"
    assert type(m.work_history[0]).__name__ == "Resume_work_history_item"
    assert m.work_history[0].start_date is None


def test_required_field_missing_raises():
    M = build_dynamic_model("Resume", SPEC)
    with pytest.raises(Exception):
        M(years_experience=5, skills=[])  # missing required candidate_name


def test_lax_schema_optional_not_required_and_has_default():
    schema = build_dynamic_model("Resume", SPEC).model_json_schema()
    assert "phone" not in schema.get("required", [])
    assert "candidate_name" in schema["required"]
    assert schema["properties"]["phone"].get("default", "MISSING") is None


def test_strict_schema_all_required_no_default_extra_forbid():
    M = build_dynamic_model("Resume", SPEC, strict=True)
    schema = M.model_json_schema()
    # Every top-level field is required under strict.
    assert set(schema["required"]) == {f["name"] for f in SPEC}
    # No implicit extra properties.
    assert schema.get("additionalProperties") is False
    # Optional field carries no 'default' key but is still nullable.
    assert "default" not in schema["properties"]["phone"]


def test_strip_defaults_removes_default_recursively():
    schema = build_dynamic_model("Resume", SPEC).model_json_schema()
    cleaned = strip_defaults(schema)

    def has_default(node):
        if isinstance(node, dict):
            return "default" in node or any(has_default(v) for v in node.values())
        if isinstance(node, list):
            return any(has_default(v) for v in node)
        return False

    assert has_default(schema)  # lax schema had defaults
    assert not has_default(cleaned)  # all stripped


SPEC_WITH_DESC = [
    {"name": "phone", "type": "string", "required": False,
     "description": "Primary contact phone number"},
    {"name": "work_history", "type": "array", "items": "object", "required": False,
     "description": "Employment entries",
     "properties": [
         {"name": "company", "type": "string", "required": True},
         {"name": "title", "type": "string", "required": True, "description": "Role title"},
     ]},
]


def test_render_field_guide_includes_names_types_desc_and_nested():
    guide = render_field_guide(SPEC_WITH_DESC)
    assert "- phone (string): Primary contact phone number" in guide
    assert "- work_history (array<object>): Employment entries" in guide
    # Nested props rendered and indented, with required marker + description.
    assert "· company (string, required)" in guide
    assert "· title (string, required): Role title" in guide


OBJECT_SPEC = [
    {"name": "candidate_name", "type": "string", "required": True},
    {
        "name": "years_experience",
        "type": "object",
        "required": True,
        "description": "Total professional experience",
        "properties": [
            {"name": "years", "type": "integer", "required": True},
            {"name": "months", "type": "integer", "required": True,
             "description": "Additional months (0-11)"},
        ],
    },
]


def test_object_field_builds_and_validates():
    M = build_dynamic_model("Resume", OBJECT_SPEC)
    m = M(candidate_name="Jane", years_experience={"years": 1, "months": 10})
    assert m.years_experience.years == 1
    assert m.years_experience.months == 10


def test_field_guide_renders_object_field_and_nested_props():
    guide = render_field_guide(OBJECT_SPEC)
    assert "- years_experience (object, required): Total professional experience" in guide
    assert "· years (integer, required)" in guide
    assert "· months (integer, required): Additional months (0-11)" in guide


def test_order_by_spec_orders_and_appends_extras():
    spec = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    result = {"c": 3, "extra": 9, "a": 1}  # b missing, extra not in spec
    out = order_by_spec(result, spec)
    # Spec fields present come first in spec order, then extras.
    assert list(out.keys()) == ["a", "c", "extra"]
    assert out == {"a": 1, "c": 3, "extra": 9}


def test_build_review_model_wraps_resume_with_approved():
    resume = build_dynamic_model("Resume", SPEC)
    Review = build_review_model(resume)
    inst = Review(
        approved=False,
        corrected={
            "candidate_name": "Jane",
            "years_experience": 5,
            "skills": ["python"],
            "work_history": [{"company": "Acme", "title": "SWE"}],
        },
    )
    assert inst.approved is False
    assert inst.corrected.candidate_name == "Jane"
    assert inst.corrected.work_history[0].company == "Acme"
