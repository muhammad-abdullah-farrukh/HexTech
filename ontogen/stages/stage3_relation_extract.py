"""Stage 3 — Relation Extraction (verbatim prompt from paper).

Fix log:
- SUBJECT TAGGING compatibility (root cause #1 follow-up): Stage 1 now
  emits CQs as list[dict] ({"subject": ..., "question": ...}) instead of
  list[str] — see stage1_cq_gen.py's fix log. This stage previously built
  its cq_block by directly f-string-interpolating each cq
  (f"CQ{i+1}. {q}"), which would silently stringify a dict (e.g.
  "CQ1. {'subject': 'person', 'question': '...'}") instead of raising an
  error — a silent prompt-corruption bug, not a crash, and one that
  would have gone unnoticed without specifically checking this file.
  _cq_text() below extracts the plain question string regardless of
  whether the item is the new dict shape or a legacy plain string, so
  the LLM always sees clean question text either way.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm
from config import OUTPUTS_DIR

PROMPT_TEMPLATE = """\
You are an assistant in building a knowledge graph. Analyze the
following competency questions and identify all
relationships mentioned in the questions.

Extract each relation first, then describe the usage of each
relation based on your understanding given the context of
competency questions.

You should only extract properties between entities and
literals, not entities themselves, or classes of entities.
Therefore, not all CQs contain valid properties.

If you don't know the answer, just say that you don't know, don't
try to make up an answer.

List all relations found.

Do not reply using a complete sentence, and only give the
answer in the following format.

Below are the examples and follow the same format to extract
the relations:

####
Document: Douglas Noel Adams (11 March 1952 − 11 May 2001) was
an English author, humourist, and screenwriter...
####
Questions:
CQ1. What is the date of birth of Douglas Noel Adams?
CQ2. What is the date of death of Douglas Noel Adams?
CQ3. What is the occupation of Douglas Noel Adams?
CQ4. What is the country of citizenship of Douglas Noel Adams?
CQ5. What is the most notable work of Douglas Noel Adams?
CQ6. What is the original medium of The Hitchhiker's Guide to the Galaxy?
CQ7. In what year was The Hitchhiker's Guide to the Galaxy originally broadcast?
CQ8. How many books are in The Hitchhiker's Guide to the Galaxy "trilogy"?
CQ9. What other media adaptations were created based on The Hitchhiker's Guide to the Galaxy?
####
Relations:
(date of birth, The date on which the subject was born.)
(date of death, The date on which the subject died.)
(occupation, The occupation of a person.)
(country of citizenship, The country of which the subject is a citizen.)
(notable work, The most notable work of a person.)
(genre, The genre or type of work.)
(publication date, The date or period when a work was first published or released.)
(has part, Indicates that the subject has a certain part, component, or element.)
(series, Indicates that the subject is part of a series, such as a book series, film series, or television series.)
####
Document:
{document}
####
Questions:
{cqs}
####
Relations:
"""

# matches (property name, description)
_REL_RE = re.compile(r"\(\s*([^,]+?)\s*,\s*(.+?)\s*\)", re.DOTALL)


def _cq_text(cq) -> str:
    """Extract the plain question string regardless of CQ shape — the
    current {"subject": ..., "question": ...} dict, or a legacy plain
    string. Never lets a dict's repr() leak into the prompt."""
    if isinstance(cq, dict):
        return str(cq.get("question", "")).strip()
    return str(cq).strip()


def extract_relations(doc_text: str, cqs: list) -> list[dict]:
    questions = [_cq_text(cq) for cq in cqs]
    cq_block = "\n".join(f"CQ{i+1}. {q}" for i, q in enumerate(questions) if q)
    prompt = PROMPT_TEMPLATE.format(document=doc_text, cqs=cq_block)
    raw = call_llm(prompt)
    relations = []
    for m in _REL_RE.finditer(raw):
        relations.append({"property": m.group(1), "description": m.group(2)})
    return relations


def run(doc_name: str, doc_text: str, cqs: list) -> list[dict]:
    relations = extract_relations(doc_text, cqs)
    out = OUTPUTS_DIR / "relations" / f"{doc_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(relations, indent=2))
    print(f"[Stage 3] {len(relations)} relations → {out}")
    return relations