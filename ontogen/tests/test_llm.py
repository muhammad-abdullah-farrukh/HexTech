"""Unit tests for call_llm_answer's truncation contract (no server needed).

The Path B starvation bug was: a reasoning model's <think> block consumed a tiny
max_tokens, _strip_think returned "", and the stage read "" as a negative answer.
call_llm_answer exists to make truncation a distinct, retried, flaggable event —
these tests lock that contract so a future edit can't quietly reintroduce
"truncated == no".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ontogen root

import stages.llm as llm
from stages.llm import call_llm_answer


def _stub_call_llm(returns):
    """Stub for call_llm yielding successive (content, finish_reason) tuples and
    recording the max_tokens each call was made with."""
    state = {"n": 0, "budgets": []}

    def _stub(prompt, max_tokens=None, return_finish_reason=False, **kw):
        state["budgets"].append(max_tokens)
        content, finish = returns[state["n"]]
        state["n"] += 1
        return (content, finish) if return_finish_reason else content

    return _stub, state


def test_clean_answer_not_truncated(monkeypatch):
    stub, state = _stub_call_llm([("yes 87", "stop")])
    monkeypatch.setattr(llm, "call_llm", stub)

    content, truncated = call_llm_answer("p", 8192, retry_budget=16384)

    assert content == "yes 87"
    assert truncated is False
    assert state["n"] == 1                    # a clean answer must not retry
    assert state["budgets"] == [8192]


def test_length_finish_retries_then_succeeds(monkeypatch):
    stub, state = _stub_call_llm([("", "length"), ("no 12", "stop")])
    monkeypatch.setattr(llm, "call_llm", stub)

    content, truncated = call_llm_answer("p", 8192, retry_budget=16384)

    assert truncated is False                 # the retry produced a real answer
    assert content == "no 12"
    assert state["budgets"] == [8192, 16384]  # retry used the larger budget


def test_empty_content_counts_as_truncated(monkeypatch):
    stub, state = _stub_call_llm([("", "stop"), ("", "stop")])
    monkeypatch.setattr(llm, "call_llm", stub)

    content, truncated = call_llm_answer("p", 8192, retry_budget=16384)

    assert truncated is True                  # empty after <think>-strip == no answer
    assert state["n"] == 2                     # empty still triggers the one retry


def test_still_truncated_after_retry_is_indeterminate(monkeypatch):
    stub, state = _stub_call_llm([("", "length"), ("", "length")])
    monkeypatch.setattr(llm, "call_llm", stub)

    content, truncated = call_llm_answer("p", 8192, retry_budget=16384)

    # The whole point: this must be reported as truncated so the caller flags it,
    # NOT returned as an answer the caller would parse into a "no".
    assert truncated is True


def test_no_retry_when_budget_not_larger(monkeypatch):
    stub, state = _stub_call_llm([("", "length")])
    monkeypatch.setattr(llm, "call_llm", stub)

    content, truncated = call_llm_answer("p", 16384, retry_budget=16384)

    assert truncated is True
    assert state["n"] == 1                     # retry_budget not > max_tokens → no retry
