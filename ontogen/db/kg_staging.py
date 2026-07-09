"""Stage KG nodes/edges into graph_entities / graph_relationships before Neo4j.

Two writers land in the same tables — the shared merge point where Path A
(deterministic, from structured fields) and Path B (LLM, from free text)
converge:

  - stage_graph()               — Path B: parse Stage 9's Turtle
                                   (literal-object → node property, uri-object → edge).
  - stage_structured_relations() — Path A: write concrete triples straight from
                                   structured_to_relations() output, entity-resolving
                                   only the objects (no Stage 9 LLM involved).

Within-run duplicate URIs across the two paths are fine: the Neo4j loader MERGEs
on properties->>'uri', so they collapse there.
"""
from __future__ import annotations

import re

from sqlalchemy.orm import Session

from db.models import GraphEntity, GraphRelationship

WD_NS = "http://www.wikidata.org/entity/"
WDT_NS = "http://www.wikidata.org/prop/direct/"


# ── helpers (mirror graphdb/load_to_neo4j.py's conventions) ──────────────────

def _local_name(uri: str, ns: str) -> str:
    return uri[len(ns):] if uri.startswith(ns) else uri


def _rel_type(predicate_local: str) -> str:
    """'worksAt' -> 'WORKS_AT'  (camel/snake -> SCREAMING_SNAKE)."""
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", predicate_local)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return s.strip("_").upper() or "RELATED_TO"


def _slugify(name: str) -> str:
    # Lowercased so differently-cased mentions of the same real-world entity
    # (e.g. "Scikit-learn" vs "scikit-learn") always collapse onto one URI —
    # canonicalization alone can't guarantee consistent casing (gazetteer
    # gaps, free-form Tier-3 LLM output), so this is the one choke point
    # every non-QID entity URI passes through before hitting Neo4j.
    slug = re.sub(r"[^\w]+", "_", (name or "").strip().lower())
    return re.sub(r"_+", "_", slug).strip("_") or "unknown"


def _add_literal(node: dict, key: str, val: str) -> None:
    """Attach a literal to a node's properties, collapsing repeats into a list
    (same behaviour as load_to_neo4j.parse_ttl)."""
    existing = node.get(key)
    if existing is None:
        node[key] = val
    elif isinstance(existing, list):
        if val not in existing:
            existing.append(val)
    elif existing != val:
        node[key] = [existing, val]


# ── primitives ───────────────────────────────────────────────────────────────

def stage_entity(session: Session, source_doc, entity_type: str, properties: dict) -> str:
    """Insert one graph_entities row; return its id (str). `properties` must
    include a 'uri' key (the dedup/MERGE key used downstream)."""
    ent = GraphEntity(entity_type=entity_type, properties=properties, source_doc=source_doc)
    session.add(ent)
    session.flush()  # populate ent.id without committing
    return str(ent.id)


def stage_relationship(
    session: Session, source_doc, from_id: str, to_id: str, rel_type: str, properties: dict | None = None
) -> str:
    """Insert one graph_relationships row; return its id (str)."""
    rel = GraphRelationship(
        from_entity=from_id,
        to_entity=to_id,
        rel_type=rel_type,
        properties=properties,
        source_doc=source_doc,
    )
    session.add(rel)
    session.flush()
    return str(rel.id)


def _persist(session: Session, source_doc, nodes: dict[str, dict], edges: list[tuple[str, str, str]]) -> None:
    """Write a {uri: props} node map and (subj_uri, rel_type, obj_uri) edges,
    resolving edge endpoints through the node id map. Commits once."""
    uri_to_id: dict[str, str] = {}
    for uri, props in nodes.items():
        etype = props.pop("_type", "Entity")
        props.setdefault("uri", uri)
        uri_to_id[uri] = stage_entity(session, source_doc, etype, props)

    for subj_uri, rel_type, obj_uri in edges:
        # Endpoints must exist as nodes; skip dangling edges rather than crash.
        if subj_uri in uri_to_id and obj_uri in uri_to_id:
            stage_relationship(
                session, source_doc, uri_to_id[subj_uri], uri_to_id[obj_uri], rel_type
            )
    session.commit()


# ── Path B: Stage 9 Turtle → staging ─────────────────────────────────────────

def stage_graph(session: Session, source_doc, turtle_str: str) -> None:
    """Parse Stage 9's Turtle and stage its nodes/edges (Path B)."""
    import rdflib

    g = rdflib.Graph()
    g.parse(data=turtle_str, format="turtle")
    rdfs_label = str(rdflib.RDFS.label)

    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str, str]] = []

    def ensure(uri: str):
        nodes.setdefault(uri, {"uri": uri, "name": _local_name(uri, WD_NS).replace("_", " "), "_type": "Entity"})

    for s, p, o in g:
        s_str, p_str = str(s), str(p)
        if not s_str.startswith(WD_NS):
            continue
        if p_str == rdfs_label:
            ensure(s_str)
            nodes[s_str]["name"] = str(o)
            continue
        if not p_str.startswith(WDT_NS):
            continue
        pred_local = _local_name(p_str, WDT_NS)
        ensure(s_str)
        if isinstance(o, rdflib.Literal):
            _add_literal(nodes[s_str], pred_local, str(o))
        else:
            o_str = str(o)
            ensure(o_str)
            edges.append((s_str, _rel_type(pred_local), o_str))

    _persist(session, source_doc, nodes, edges)


# ── Path A: structured relations → staging (no Stage 9 LLM) ──────────────────

def stage_structured_relations(session: Session, source_doc, relations: list[dict], resolver=None) -> None:
    """Write structured_to_relations() output straight to the staging tables.

    - literal objects become properties on the subject node (so the Person node
      carries email/phone/years, dates/grades attach to their subject);
    - entity objects are entity-resolved (via `resolver`, if given) and become
      nodes + edges. `resolver.resolve(mention, entity_type)` follows the same
      Tier 1/2/3 contract as ResumeEntityResolver.
    """
    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str, str]] = []

    def ensure(uri: str, name: str, etype: str) -> None:
        node = nodes.setdefault(uri, {"uri": uri, "name": name, "_type": etype})
        # A later, more specific type wins over the generic default.
        if etype not in ("Entity", "unknown"):
            node["_type"] = etype

    for rel in relations:
        subject = str(rel.get("subject", "")).strip()
        obj = str(rel.get("object", "")).strip()
        prop = str(rel.get("property", "")).strip()
        if not subject or not obj or not prop:
            continue

        subj_type = str(rel.get("subject_type") or "person").strip()
        subj_uri = WD_NS + _slugify(subject)
        ensure(subj_uri, subject, subj_type)

        if str(rel.get("object_type", "entity")).lower() == "literal":
            _add_literal(nodes[subj_uri], prop, obj)
            continue

        # Entity object — resolve, then build node + edge.
        obj_etype = str(rel.get("object_entity_type") or "unknown").strip()
        canonical, qid = obj, None
        if resolver is not None:
            try:
                res = resolver.resolve(obj, obj_etype)
                canonical = res.canonical_form or obj
                qid = res.wikidata_qid
            except Exception:
                canonical = obj
        obj_uri = WD_NS + (qid if qid else _slugify(canonical))
        ensure(obj_uri, canonical, obj_etype)
        edges.append((subj_uri, _rel_type(prop), obj_uri))

    _persist(session, source_doc, nodes, edges)
