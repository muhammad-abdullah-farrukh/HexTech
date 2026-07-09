"""Tests for stages/llm.py's per-attempt call logging and model-mismatch
detection (see that module's docstring for why: LLM_BASE_URL can point at a
load-balanced endpoint that silently answers a request with a different
model than config.LLM_MODEL names — observed live 2026-07-09 on the cluster
endpoint, ~20% of calls answered by a different backend/model with zero
client-visible signal besides the response body's own "model" field).

All HTTP is mocked — no live network dependency.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

import stages.llm as llm


def _mock_response(model: str, content: str = "yes 90",
                    finish_reason: str = "stop", completion_tokens: int = 5):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": completion_tokens},
    }
    return resp


def _read_log(log_path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    log_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(llm, "_CALL_LOG_PATH", log_path)
    return log_path


def test_success_logs_served_model_and_no_mismatch_when_equal(isolated_log):
    with patch.object(llm.requests, "post", return_value=_mock_response(llm.LLM_MODEL)):
        content = llm.call_llm("hello")

    assert content == "yes 90"
    entries = _read_log(isolated_log)
    assert len(entries) == 1
    e = entries[0]
    assert e["outcome"] == "success"
    assert e["requested_model"] == llm.LLM_MODEL
    assert e["served_model"] == llm.LLM_MODEL
    assert e["model_mismatch"] is False
    assert e["completion_tokens"] == 5
    assert e["finish_reason"] == "stop"


def test_mismatch_detected_and_logged(isolated_log, capsys):
    with patch.object(llm.requests, "post", return_value=_mock_response("some-other-model")):
        llm.call_llm("hello")

    entries = _read_log(isolated_log)
    assert entries[0]["served_model"] == "some-other-model"
    assert entries[0]["model_mismatch"] is True
    # Live-visible warning, not just a buried log line.
    assert "MODEL MISMATCH" in capsys.readouterr().out


def test_missing_model_field_is_not_treated_as_a_match(isolated_log):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": "yes 90"}, "finish_reason": "stop"}],
    }
    with patch.object(llm.requests, "post", return_value=resp):
        llm.call_llm("hello")

    entries = _read_log(isolated_log)
    assert entries[0]["served_model"] == ""
    assert entries[0]["model_mismatch"] is False  # absent field: unknown, not a false mismatch


def test_connection_error_logs_every_retry_attempt_then_raises(isolated_log, monkeypatch):
    monkeypatch.setattr(llm, "MAX_RETRIES", 2)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)  # skip real backoff

    with patch.object(
        llm.requests, "post",
        side_effect=llm.requests.exceptions.ConnectionError("refused"),
    ):
        with pytest.raises(llm.requests.exceptions.ConnectionError):
            llm.call_llm("hello")

    entries = _read_log(isolated_log)
    assert len(entries) == 2  # one per attempt
    assert all(e["outcome"] == "ConnectionError" for e in entries)
    assert [e["attempt"] for e in entries] == [1, 2]
    # Failure-path entries never claim a served model.
    assert all("served_model" not in e for e in entries)


def test_endpoint_override_uses_that_base_url_and_model(isolated_log):
    ep = {"base_url": "http://example:1234/v1", "model": "some-model"}
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["model"] = json["model"]
        return _mock_response("some-model")

    with patch.object(llm.requests, "post", side_effect=_fake_post):
        llm.call_llm("hello", endpoint=ep)

    assert captured["url"] == "http://example:1234/v1/chat/completions"
    assert captured["model"] == "some-model"
    e = _read_log(isolated_log)[0]
    assert e["base_url"] == "http://example:1234/v1"
    assert e["requested_model"] == "some-model"
    assert e["model_mismatch"] is False  # served == requested for this endpoint


def test_logging_failure_never_breaks_a_successful_call(isolated_log, monkeypatch):
    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _boom)
    with patch.object(llm.requests, "post", return_value=_mock_response(llm.LLM_MODEL)):
        content = llm.call_llm("hello")  # must not raise despite logging failing

    assert content == "yes 90"
