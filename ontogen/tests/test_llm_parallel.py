"""Tests for stages/llm.py:call_llm_parallel() — the concurrent fan-out that
dispatches independent LLM calls across the configured endpoints.

Correctness properties pinned here:
  - results come back in the SAME order as the input tasks (callers map them
    back positionally, e.g. qa_pairs[i] ↔ cqs[i]);
  - every task runs exactly once across the workers (no drops, no dupes);
  - work actually spreads across >1 endpoint;
  - a task whose assigned endpoint always fails fails OVER to another endpoint;
  - on_error="raise" surfaces a hard failure; on_error="return" keeps it inline.

call_llm itself is monkeypatched — no HTTP, no live network.
"""
import threading

import pytest

import stages.llm as llm


TWO_ENDPOINTS = [
    {"base_url": "http://a/v1", "model": "model-a", "slots": 2},
    {"base_url": "http://b/v1", "model": "model-b", "slots": 2},
]


def test_empty_tasks_returns_empty():
    assert llm.call_llm_parallel([], endpoints=TWO_ENDPOINTS) == []


def test_results_preserve_input_order(monkeypatch):
    def fake_call(prompt, endpoint=None, **kw):
        return f"answer:{prompt}"

    monkeypatch.setattr(llm, "call_llm", fake_call)
    tasks = [{"prompt": f"q{i}"} for i in range(20)]
    out = llm.call_llm_parallel(tasks, endpoints=TWO_ENDPOINTS)
    assert out == [f"answer:q{i}" for i in range(20)]


def test_every_task_runs_exactly_once(monkeypatch):
    seen = []
    lock = threading.Lock()

    def fake_call(prompt, endpoint=None, **kw):
        with lock:
            seen.append(prompt)
        return prompt

    monkeypatch.setattr(llm, "call_llm", fake_call)
    tasks = [{"prompt": f"q{i}"} for i in range(30)]
    llm.call_llm_parallel(tasks, endpoints=TWO_ENDPOINTS)
    assert sorted(seen) == sorted(t["prompt"] for t in tasks)
    assert len(seen) == 30  # no duplicates from failover/requeue


def test_work_spreads_across_both_endpoints(monkeypatch):
    used = []
    lock = threading.Lock()

    def fake_call(prompt, endpoint=None, **kw):
        with lock:
            used.append(endpoint["base_url"])
        # a little work so threads actually overlap
        import time
        time.sleep(0.01)
        return prompt

    monkeypatch.setattr(llm, "call_llm", fake_call)
    tasks = [{"prompt": f"q{i}"} for i in range(40)]
    llm.call_llm_parallel(tasks, endpoints=TWO_ENDPOINTS)
    assert set(used) == {"http://a/v1", "http://b/v1"}


def test_failover_to_other_endpoint_when_assigned_one_fails(monkeypatch):
    # model-a always fails; model-b always works. Every task must still succeed
    # by failing over to b.
    def fake_call(prompt, endpoint=None, **kw):
        if endpoint["model"] == "model-a":
            raise RuntimeError("endpoint a down")
        return f"b:{prompt}"

    monkeypatch.setattr(llm, "call_llm", fake_call)
    tasks = [{"prompt": f"q{i}"} for i in range(10)]
    out = llm.call_llm_parallel(tasks, endpoints=TWO_ENDPOINTS)
    assert out == [f"b:q{i}" for i in range(10)]


def test_on_error_raise_surfaces_hard_failure(monkeypatch):
    def fake_call(prompt, endpoint=None, **kw):
        raise ValueError(f"boom {prompt}")

    monkeypatch.setattr(llm, "call_llm", fake_call)
    tasks = [{"prompt": "q0"}, {"prompt": "q1"}]
    with pytest.raises(ValueError):
        llm.call_llm_parallel(tasks, endpoints=TWO_ENDPOINTS, on_error="raise")


def test_on_error_return_keeps_exception_inline(monkeypatch):
    def fake_call(prompt, endpoint=None, **kw):
        if prompt == "bad":
            raise ValueError("boom")
        return prompt

    monkeypatch.setattr(llm, "call_llm", fake_call)
    tasks = [{"prompt": "ok0"}, {"prompt": "bad"}, {"prompt": "ok1"}]
    out = llm.call_llm_parallel(tasks, endpoints=TWO_ENDPOINTS, on_error="return")
    assert out[0] == "ok0"
    assert isinstance(out[1], ValueError)
    assert out[2] == "ok1"


def test_endpoint_key_in_task_is_not_forwarded_as_kwarg(monkeypatch):
    # A stray "endpoint" key in a task dict must be stripped, not passed
    # through as a duplicate kwarg (which would TypeError).
    def fake_call(prompt, endpoint=None, **kw):
        assert "endpoint" not in kw
        return prompt

    monkeypatch.setattr(llm, "call_llm", fake_call)
    tasks = [{"prompt": "q0", "endpoint": {"bogus": True}}]
    out = llm.call_llm_parallel(tasks, endpoints=TWO_ENDPOINTS)
    assert out == ["q0"]


def test_single_endpoint_still_runs_all(monkeypatch):
    def fake_call(prompt, endpoint=None, **kw):
        return prompt

    monkeypatch.setattr(llm, "call_llm", fake_call)
    one = [{"base_url": "http://a/v1", "model": "model-a", "slots": 1}]
    tasks = [{"prompt": f"q{i}"} for i in range(5)]
    out = llm.call_llm_parallel(tasks, endpoints=one)
    assert out == [f"q{i}" for i in range(5)]
