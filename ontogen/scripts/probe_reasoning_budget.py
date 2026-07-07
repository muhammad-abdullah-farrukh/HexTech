"""Throwaway probe: measure deepseek-r1's reasoning-length distribution for the
Path B classification/definition prompts, so config.LLM_TOKENS_* are set from data
instead of guessed.

Why this exists: the Path B stages were capped at max_tokens=10/40/60/300, sized for
a non-reasoning model's answer length. deepseek-r1 emits a <think>…</think> block
BEFORE the answer, so those caps were consumed entirely by reasoning and the answer
never started — the root cause of Path B producing zero facts. The current stage logs
can't tell us how long the reasoning actually runs (every sample was truncated at the
old caps), so we can't read the distribution off them. This script runs the real
prompt templates with a large budget and reports how many tokens the model actually
generates (reasoning + answer), read from the API's usage.completion_tokens — the true
generated length, which len(content) after <think>-stripping would hide.

Run (venv active, DATABASE_URL not required — this only hits the LLM endpoint):
    python ontogen/scripts/probe_reasoning_budget.py
    python ontogen/scripts/probe_reasoning_budget.py --reps 5 --max-tokens 16384

Set config.LLM_TOKENS_CLASSIFY / LLM_TOKENS_DEFINE from the printed
`recommended (ceil(p99*1.25), floored 8192)` line, and if p99 lands near the probe's
max-tokens the window itself is the limit — curtail thinking server-side instead of
raising further. Keep the printed table with the commit as the justification.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]  # ontogen/
sys.path.insert(0, str(ROOT))

from config import LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_TEMPERATURE
from stages.stage6_match_validate import VALIDATE_PROMPT
from stages.canonicalize import _VERIFY_PROMPT, _DEFINE_PROMPT, _ENTITY_LLM_PROMPT

# ── Representative, already-filled prompts (mirror real Path B call shapes) ──

_VALIDATE = [
    VALIDATE_PROMPT.format(p1=p1, p2=p2)
    for p1, p2 in [
        ("email: The email address of the person.",
         "Addressee: person or organization to whom a letter is addressed"),
        ("hasSkill: A skill the person has (Reinforcement Learning).",
         "field of work: field of work of a person or organization"),
        ("educatedAt: An institution the person was educated at (GIKI).",
         "educated at: educational institution attended by a person"),
        ("yearsExperience: Total professional experience of the person.",
         "work period (start): start of period during which a person held a position"),
        ("speaksLanguage: A human language the person knows (Urdu).",
         "languages spoken, written or signed: language a person uses"),
    ]
]

_VERIFY = [
    _VERIFY_PROMPT.format(label_a=la, definition_a=da, label_b=lb, definition_b=db)
    for la, da, lb, db in [
        ("employer", "The organization a person is employed by.",
         "worksAt", "The organization where a person performs their job."),
        ("hasSkill", "A competency or technology a person is proficient in.",
         "educatedAt", "An institution where a person studied."),
        ("speaksLanguage", "A natural language a person can communicate in.",
         "languageOfWork", "A language a person uses professionally."),
    ]
]

_DEFINE = [
    _DEFINE_PROMPT.format(label=label, description=desc, cqs="- " + cq)
    for label, desc, cq in [
        ("achievesMetric", "A quantitative result reported for a project.",
         "What accuracy did the project reach?"),
        ("participatedIn", "An activity or membership of the person.",
         "Which societies is the person a member of?"),
        ("hasReference", "A person listed as a reference for the candidate.",
         "Who are the candidate's references?"),
    ]
]

_ENTITY = [
    _ENTITY_LLM_PROMPT.format(mention=m, entity_type=t, context=c or "N/A")
    for m, t, c in [
        ("Google LLC", "company", ""),
        ("MIT", "university", "graduate school"),
        ("AWS Certified Solutions Architect - Associate", "certification", ""),
        ("Sr. SWE", "job_title", "backend team"),
    ]
]

TEMPLATES = {
    "stage6_validate": _VALIDATE,
    "edc_verify":      _VERIFY,
    "edc_define":      _DEFINE,
    "entity_tier3":    _ENTITY,
}


def _post(prompt: str, max_tokens: int) -> tuple[int, str]:
    """POST directly (not via call_llm — we need usage.completion_tokens and the
    raw finish_reason, both of which call_llm discards). Returns
    (completion_tokens, finish_reason)."""
    resp = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    # completion_tokens is the true generated length (reasoning + answer). Fall
    # back to a rough estimate only if the server omits usage.
    ct = usage.get("completion_tokens")
    if ct is None:
        ct = len(choice["message"].get("content") or "") // 3
    return int(ct), choice.get("finish_reason", "stop")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="calls per prompt")
    ap.add_argument("--max-tokens", type=int, default=16384)
    args = ap.parse_args()

    print(f"endpoint={LLM_BASE_URL} model={LLM_MODEL} "
          f"reps={args.reps} max_tokens={args.max_tokens}\n", flush=True)

    all_classify: list[int] = []   # stage6_validate + edc_verify + entity_tier3
    all_define: list[int] = []     # edc_define
    any_length_hit = False

    print(f"{'template':<18}{'n':>4}{'min':>8}{'mean':>8}{'p95':>8}"
          f"{'p99':>8}{'max':>8}{'len_hits':>10}")
    print("-" * 74)

    for name, prompts in TEMPLATES.items():
        samples: list[int] = []
        length_hits = 0
        for prompt in prompts:
            for _ in range(args.reps):
                t0 = time.time()
                ct, finish = _post(prompt, args.max_tokens)
                samples.append(ct)
                if finish == "length":
                    length_hits += 1
                    any_length_hit = True
                print(f"  [{name}] {ct:>6} tok  finish={finish:<6} "
                      f"{time.time()-t0:>5.1f}s", flush=True)

        arr = np.array(samples)
        print(f"{name:<18}{len(arr):>4}{arr.min():>8}{arr.mean():>8.0f}"
              f"{np.percentile(arr, 95):>8.0f}{np.percentile(arr, 99):>8.0f}"
              f"{arr.max():>8}{length_hits:>10}")

        (all_define if name == "edc_define" else all_classify).extend(samples)

    def _recommend(samples: list[int]) -> int:
        p99 = float(np.percentile(np.array(samples), 99))
        return max(8192, int(math.ceil(p99 * 1.25 / 256) * 256))  # round up to 256

    print("\nRecommended config values (ceil(p99 * 1.25), rounded to 256, floored 8192):")
    print(f"  LLM_TOKENS_CLASSIFY = {_recommend(all_classify)}")
    print(f"  LLM_TOKENS_DEFINE   = {_recommend(all_define)}")
    biggest = max(_recommend(all_classify), _recommend(all_define))
    print(f"  LLM_TOKENS_RETRY    >= {max(16384, 2 * biggest)}   "
          f"(keep meaningful headroom over the base budgets)")

    if any_length_hit:
        print("\n⚠ Some samples still hit finish_reason='length' at "
              f"max_tokens={args.max_tokens}. The context window itself is the limit "
              "for those prompts — curtail deepseek-r1 thinking server-side "
              "(/no_think or chat-template flag) rather than raising the budget.")


if __name__ == "__main__":
    main()
