"""Dynamic Pydantic model construction from a runtime field-spec.

The target schema arrives at runtime as a list of field dicts (see
config/field_spec.json), not a hardcoded class. `build_dynamic_model` recursively
compiles it to nested `BaseModel` classes so object-arrays become real validated
types (e.g. `work_history: List[WorkHistoryItem]`).

Two emission modes are supported because hosted structured-output providers vary:

  * lax (default)  — optional fields use `Optional[X] = None`. Pydantic v2 emits
    these as `{"anyOf": [..., {"type": "null"}], "default": null}`. Works with
    instructor's default JSON_SCHEMA assumptions and lenient providers.

  * strict         — every field is `required` (keys must be present) but optional
    ones stay nullable (`Optional[X]` with no default), and `extra="forbid"` makes
    the emitted schema carry `additionalProperties: false`. This matches
    OpenAI-strict-style providers.

Independently, `strip_defaults` removes every `"default"` key from a JSON schema
dict. Some providers reject any schema containing `default` regardless of the
required/additionalProperties strictness — that is a *different* failure from a
field being dropped for not being required, and it has a different fix (strip
defaults before sending, no model change). See the plan, §4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


def load_field_spec(path: str | Path) -> list[dict]:
    """Load a field-spec list from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"field_spec at {path} must be a JSON list of field dicts.")
    return data


def build_dynamic_model(name: str, fields: list[dict], strict: bool = False):
    """Recursively build a Pydantic model from a field-spec list.

    When `strict` is True, optional fields remain required keys (nullable values)
    and the model forbids extra properties — producing an OpenAI-strict-compatible
    JSON schema.
    """
    field_defs: dict[str, Any] = {}
    for f in fields:
        if f["type"] == "object" and "properties" in f:
            py_type = build_dynamic_model(
                f"{name}_{f['name']}", f["properties"], strict=strict
            )
        elif f["type"] == "array":
            if f.get("items") == "object" and "properties" in f:
                nested = build_dynamic_model(
                    f"{name}_{f['name']}_item", f["properties"], strict=strict
                )
                py_type = List[nested]
            else:
                py_type = List[TYPE_MAP.get(f.get("items", "string"), str)]
        else:
            py_type = TYPE_MAP.get(f["type"], str)

        required = f.get("required", False)
        description = f.get("description", "")

        if required:
            field_def = (py_type, Field(..., description=description))
        elif strict:
            # Required key, but the value may be null. No default -> no "default"
            # key in the schema, and the field still appears in `required`.
            field_def = (Optional[py_type], Field(..., description=description))
        else:
            field_def = (Optional[py_type], Field(default=None, description=description))

        field_defs[f["name"]] = field_def

    model_config = ConfigDict(extra="forbid") if strict else None
    if model_config is not None:
        return create_model(name, __config__=model_config, **field_defs)
    return create_model(name, **field_defs)


def strip_defaults(schema: Any) -> Any:
    """Recursively remove every `"default"` key from a JSON-schema dict.

    Use when a provider rejects schemas that contain `default` (distinct from
    required/additionalProperties strictness).
    """
    if isinstance(schema, dict):
        return {
            k: strip_defaults(v) for k, v in schema.items() if k != "default"
        }
    if isinstance(schema, list):
        return [strip_defaults(v) for v in schema]
    return schema


def _field_type_label(f: dict) -> str:
    """Human-readable type label for one field-spec entry, e.g. 'array<object>'."""
    if f["type"] == "array":
        items = f.get("items", "string")
        return f"array<{items}>"
    return f["type"]


def render_field_guide(field_spec: list[dict]) -> str:
    """Render field names + types + required + descriptions as a compact guide.

    Grammar-constrained decoding forces the model to emit the schema's field
    *names*, but the field *descriptions* never reach it. Injecting this guide into
    the prompt restores that semantic information (e.g. what `phone` or `projects`
    should contain, including nested object properties).
    """
    lines: list[str] = []
    for f in field_spec:
        req = ", required" if f.get("required", False) else ""
        desc = f.get("description", "").strip()
        desc_part = f": {desc}" if desc else ""
        lines.append(f"- {f['name']} ({_field_type_label(f)}{req}){desc_part}")
        # Nested object properties (for object fields or array<object> fields).
        for prop in f.get("properties", []):
            preq = ", required" if prop.get("required", False) else ""
            pdesc = prop.get("description", "").strip()
            pdesc_part = f": {pdesc}" if pdesc else ""
            lines.append(
                f"    · {prop['name']} ({_field_type_label(prop)}{preq}){pdesc_part}"
            )
    return "\n".join(lines)


def order_by_spec(result: dict, field_spec: list[dict]) -> dict:
    """Reorder a result dict to match field_spec order (top-level keys).

    Spec fields come first in declaration order; any keys not in the spec are
    appended after, preserving their existing order. Fixes fields added out of
    order by post-processing (e.g. the contact backfill appending `phone`).
    """
    spec_names = [f["name"] for f in field_spec]
    ordered = {name: result[name] for name in spec_names if name in result}
    for k, v in result.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def build_review_model(resume_model: type[BaseModel]) -> type[BaseModel]:
    """Wrap a resume model for the verify/refine pass.

    Returns a model `{approved: bool, corrected: <resume_model>}` so one LLM call
    can both judge the current JSON and return a corrected/completed version.
    """
    return create_model(
        "ResumeReview",
        approved=(
            bool,
            Field(
                ...,
                description="true only if `corrected` is complete and correct "
                "versus the source text and needs no further changes",
            ),
        ),
        corrected=(
            resume_model,
            Field(..., description="the full resume: unchanged if approved, else "
                  "corrected with missing fields filled and wrong values fixed"),
        ),
    )
