"""Stage 1 — Competency Question Generation (verbatim prompt from paper).

Fix log:
- The original prompt told the model to "let the content decide the count"
  but gave no example of what failure looks like, and nothing caught it if
  the model ignored that instruction. In practice, an 8B instruct model
  will keep counting upward ("the Nth project") well past the number of
  real items in the document once it runs out of facts — observed
  generating CQs for a "thirty-first project" against a CV that lists 6.
- Added an explicit anti-padding instruction with a negative example to the
  prompt.
- Added ORDINAL_PROJECT_RE: a post-filter that detects runaway
  ordinal-counting questions ("the Nth project/job/skill/role...") and
  truncates the list once more than a small number of them appear in a
  row. This is a safety net, not a substitute for the model actually
  following the instruction — see the comment on generate_cqs for the more
  invasive two-pass fix if this band-aid isn't enough.
- call_llm() is now given an explicit max_tokens bound so a runaway
  generation can't silently produce hundreds of padding questions.
- The prompt's own example showed plain "CQ1. text" with no markdown, but
  after switching to Ollama's native /api/chat endpoint the model started
  wrapping labels in bold anyway (e.g. "**CQ1.** text"). The old parser
  matched on line.startswith("CQ") / line[0].isdigit(), which fails the
  instant a line starts with "*" — this silently dropped ALL CQs (79
  generated, 0 parsed) and caused the pipeline to skip the document
  entirely. Added an explicit "no markdown" instruction to the prompt
  (frequency reducer, not a guarantee) AND replaced the line matcher with
  CQ_LINE_RE, a regex that tolerates 0-2 leading/trailing asterisks around
  the label. Same two-layer pattern as the ordinal-padding fix above:
  prompt instruction + parser-level safety net, since local-model
  instruction-following isn't reliable enough to trust alone.

- SUBJECT TAGGING (root cause #1 follow-up — replaces regex-based section
  guessing downstream): an earlier attempt at fixing Stage 9/10's
  context-truncation problem tried to *infer* which résumé section a CQ
  belonged to by regexing over its question text (e.g. matching "... at
  X"). That worked only as well as the model's phrasing habits happened
  to cooperate, and degraded silently (an undifferentiated "_general"
  bucket) on any document phrased differently. That is not a fix, it's a
  coincidence.

  The robust version moves the decision to where the information already
  exists: the model already "knows" which entity each question is about
  the moment it writes the question (it's generating CQ-by-CQ, walking
  the document section by section). So the model is now asked to STATE
  that entity explicitly, in a fixed, parseable format:

      CQ12. [SUBJECT: BSc - Air University] What is the start date of
      Abdullah Zahid's BSc program at Air University?

  This is a controlled output contract, not free text to be parsed by
  guessing — same two-layer pattern as CQ_LINE_RE below: prompt
  instruction (with a worked example) + a strict regex parser
  (CQ_LINE_RE's bracket group) that requires the exact bracket format,
  with a logged fallback (subject="_unlabeled") if the model ever drops
  the tag, so a missing tag degrades visibly instead of silently
  mis-grouping facts.

  generate_cqs() now returns list[dict] ({"subject": ..., "question":
  ...}) instead of list[str]. This is a breaking change to Stage 1's
  output shape — stage2_cq_answer.py and stage9_10_kg.py were updated to
  match (see their own fix logs).
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm
from config import OUTPUTS_DIR, CQ_SAFETY_MAX

PROMPT_TEMPLATE = """\
Write competency questions based on the abstract level concepts
in the document. Write questions that can be answered using
the document only.

Write one question for every distinct, verifiable fact in the
document (e.g. each role, employer, degree, certification, skill,
project, date, or named entity). Do not pad with redundant or
overlapping questions, and do not skip facts to keep the list
short. Short documents may need only a few questions; long,
detailed documents may need many more. Let the content decide
the count.

Once you have covered every fact in the document, STOP. Do not
keep generating questions by counting upward past the number of
items actually named (e.g. do not write "What is the name of the
fifth project" if the document only names three projects). If you
notice yourself about to ask about an Nth item that was never
named, that means you are done — stop instead of inventing one.

Every question must be tagged with the specific entity, role,
degree, certification, or project it is about, using this exact
format:

CQ<n>. [SUBJECT: <short label>] <question text>

Rules for the SUBJECT label:
- Use the SAME exact label text for every question about the same
  entity (e.g. every question about one specific job uses the same
  label, every question about one specific degree uses the same
  label), so they can be grouped later. Small wording differences
  ("NASTP" vs "NASTP Internship") will NOT be treated as the same
  label, so pick one label per entity and reuse it exactly.
- Use a short, distinguishing label, e.g. "BSc - Air University",
  "FSc - <college name>", "Job - NASTP Internship", "Job -
  Freelance ML Engineer", "Certification - <name>".
- For questions about the person directly rather than one specific
  entity (name, contact info, date of birth, current occupation
  summary, top-level identity facts), use the label "person".

Output plain text only. Do not use markdown formatting — no
asterisks, bold, headers, or backticks anywhere in the output.

Below are the examples and follow the same format when
generating competency questions:

####
Document: Douglas Noel Adams (11 March 1952 − 11 May 2001) was
an English author, humourist, and screenwriter, best known
for The Hitchhiker's Guide to the Galaxy (HHGTTG).
Originally a 1978 BBC radio comedy, The Hitchhiker's Guide
to the Galaxy developed into a "trilogy" of five books that
sold more than 15 million copies in his lifetime. It was
further developed into a television series, several stage
plays, comics, a video game, and a 2005 feature film. Adams'
s contribution to UK radio is commemorated in The Radio
Academy's Hall of Fame.
####
Questions:
CQ1. [SUBJECT: person] What is the date of birth of Douglas Noel Adams?
CQ2. [SUBJECT: person] What is the date of death of Douglas Noel Adams?
CQ3. [SUBJECT: person] What is the occupation of Douglas Noel Adams?
CQ4. [SUBJECT: person] What is the country of citizenship of Douglas Noel Adams?
CQ5. [SUBJECT: person] What is the most notable work of Douglas Noel Adams?
CQ6. [SUBJECT: The Hitchhiker's Guide to the Galaxy] What is the original medium of The Hitchhiker's Guide to
the Galaxy?
CQ7. [SUBJECT: The Hitchhiker's Guide to the Galaxy] In what year was The Hitchhiker's Guide to the Galaxy
originally broadcast?
CQ8. [SUBJECT: The Hitchhiker's Guide to the Galaxy] How many books are in The Hitchhiker's Guide to the Galaxy
"trilogy"?
CQ9. [SUBJECT: The Hitchhiker's Guide to the Galaxy] What other media adaptations were created based on The
Hitchhiker's Guide to the Galaxy?

(Note what CQ9 does NOT do: it does not continue with "CQ10. What
is the name of the second media adaptation?", "CQ11. What is the
name of the third media adaptation?" and so on — the document
only describes the adaptations as a group, so the questions stop
there instead of inventing a count that was never stated.)
####
Document:
{document}
####
Questions:
"""

# Catches runaway "the Nth <noun>" patterns (project, job, role, skill,
# certification, employer, ...) that only ever appear when the model is
# padding past real content — genuine documents don't get phrased this way
# by the prompt above.
ORDINAL_PROJECT_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
    r"|eleventh|twelfth|thirteenth|\d+(st|nd|rd|th)|twenty-\w+|thirty-\w+)\b"
    r".{0,40}\b(project|job|role|skill|certification|employer|degree)s?\b",
    re.IGNORECASE,
)

# How many ordinal-pattern CQs we tolerate before assuming the model has
# started padding and truncating the list there. Genuine resumes basically
# never need to say "the third project" by ordinal — they name things.
MAX_ORDINAL_HITS = 3

# Matches "CQ1. [SUBJECT: x] text", "1. [SUBJECT: x] text", and
# markdown-wrapped variants like "**CQ1.** [SUBJECT: x] text" — tolerates
# 0-2 asterisks around the label since local models don't reliably honor
# "no markdown" instructions. The [SUBJECT: ...] bracket is REQUIRED by
# the prompt contract but made optional here (group 1) so a line that
# drops the tag is still parsed as a CQ (falls back to "_unlabeled")
# instead of being silently dropped entirely — see _unlabeled handling
# in generate_cqs().
CQ_LINE_RE = re.compile(
    r'^\*{0,2}\s*(?:CQ\s*\d+|\d+)\s*\.\*{0,2}\s*'
    r'(?:\[SUBJECT:\s*([^\]]+?)\s*\]\s*)?'
    r'(.+)$'
)


def _parse_cqs(raw: str) -> list[dict]:
    cqs: list[dict] = []
    unlabeled_count = 0
    for line in raw.splitlines():
        line = line.strip()
        m = CQ_LINE_RE.match(line)
        if m:
            subject = (m.group(1) or "").strip()
            question = m.group(2).strip()
            if not subject:
                subject = "_unlabeled"
                unlabeled_count += 1
            cqs.append({"subject": subject, "question": question})
    if unlabeled_count:
        print(
            f"[Stage 1]   ⚠ {unlabeled_count} CQ(s) were missing a "
            f"[SUBJECT: ...] tag — the model didn't follow the format "
            f"instruction for these lines. They'll still be processed but "
            f"won't be grouped with related questions downstream.",
            flush=True,
        )
    return cqs


MAX_CONTINUATIONS = 3


def generate_cqs(doc_text: str) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(document=doc_text)
    raw, finish_reason = call_llm(prompt, max_tokens=4000, return_finish_reason=True)
    print("\n========== RAW LLM OUTPUT ==========")
    print(raw)
    print("====================================\n")

    cqs = _parse_cqs(raw)

    attempts = 0
    while finish_reason == "length" and attempts < MAX_CONTINUATIONS:
        attempts += 1
        last_subject = cqs[-1]["subject"] if cqs else "the start"
        last_question = cqs[-1]["question"] if cqs else ""
        print(
            f"[Stage 1]   ⚠ output truncated (continuation {attempts}/{MAX_CONTINUATIONS}) "
            f"— {len(cqs)} CQs so far, continuing from \"{last_subject}\"",
            flush=True,
        )

        continuation_prompt = (
            prompt + raw +
            f"\n\n(Your output was cut off. The last question you wrote was "
            f"about \"{last_subject}\": \"{last_question}\". Continue directly "
            f"from the next fact in the document — do not repeat any question "
            f"already asked above. Use the same [SUBJECT: ...] format. If you "
            f"have genuinely covered every fact already, write DONE and nothing "
            f"else.)"
        )
        raw, finish_reason = call_llm(continuation_prompt, max_tokens=4000, return_finish_reason=True)
        print("\n===== RAW LLM OUTPUT (continuation) =====")
        print(raw)
        print("==========================================\n")
        if raw.strip() == "DONE":
            break
        cqs.extend(_parse_cqs(raw))

    if not cqs:
        print(
            f"[Stage 1]   ⚠ parsed 0 CQs — check RAW LLM OUTPUT above for an "
            f"unexpected format",
            flush=True,
        )

    ordinal_hits = [
        i for i, cq in enumerate(cqs) if ORDINAL_PROJECT_RE.search(cq["question"])
    ]
    if len(ordinal_hits) > MAX_ORDINAL_HITS:
        cutoff = ordinal_hits[MAX_ORDINAL_HITS]
        dropped = len(cqs) - cutoff
        print(
            f"[Stage 1]   ⚠ dropped {dropped} likely-padded ordinal-counting "
            f"question(s) past index {cutoff}",
            flush=True,
        )
        cqs = cqs[:cutoff]

    return cqs[:CQ_SAFETY_MAX]

def run(doc_name: str, doc_text: str) -> list[dict]:
    cqs = generate_cqs(doc_text)
    out = OUTPUTS_DIR / "cqs" / f"{doc_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cqs, indent=2))
    n_subjects = len({cq["subject"] for cq in cqs})
    print(f"[Stage 1] {len(cqs)} CQs across {n_subjects} subject(s) → {out}")
    return cqs