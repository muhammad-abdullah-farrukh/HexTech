import pytest

from resume_parser.context import (
    estimate_tokens,
    fit_review_content,
    fit_user_content,
)
from resume_parser.settings import Settings


def make_settings(**overrides):
    base = dict(
        base_url="http://localhost:8080/v1",
        model="m",
        api_key="not-needed",
        instructor_mode="JSON_SCHEMA",
        max_retries=2,
        ratelimit_attempts=5,
        health_timeout=2.0,
        refine_passes=2,
        context_window=4096,
        context_safety_margin=256,
        chars_per_token=3.5,
        temperature=0.1,
        frequency_penalty=0.15,
        presence_penalty=0.0,
        max_tokens=1024,
    )
    base.update(overrides)
    return Settings(**base)


def test_estimate_tokens_ceil():
    assert estimate_tokens("", 3.5) == 0
    assert estimate_tokens("a" * 7, 3.5) == 2  # 7/3.5 = 2


def test_short_text_not_truncated():
    s = make_settings()
    text = "short resume text"
    out, info = fit_user_content("system", text, s)
    assert out == text
    assert info.truncated is False


def test_long_text_truncated_to_fit():
    s = make_settings()
    # available = 4096 - (1024 + 256) - sys_tokens. With ~3.5 chars/token that's
    # roughly 2800 tokens -> ~9800 chars. Make the resume far larger.
    text = "x" * 50000
    out, info = fit_user_content("system prompt", text, s)
    assert info.truncated is True
    assert len(out) < len(text)
    # The kept text must fit the available budget (with the same estimator).
    assert estimate_tokens(out, s.chars_per_token) <= info.available_for_user


def test_no_room_raises():
    # max_tokens + margin already exceed the window -> no room for any user text.
    s = make_settings(context_window=1000, max_tokens=1024, context_safety_margin=256)
    with pytest.raises(RuntimeError, match="No room for resume text"):
        fit_user_content("system", "anything", s)


def test_review_content_includes_json_and_source():
    s = make_settings()
    out, info = fit_review_content("sys", "the source text", '{"a": 1}', s)
    assert "the source text" in out
    assert '{"a": 1}' in out
    assert info.truncated is False
    assert info.protected_tokens > 0


def test_review_truncates_only_source_not_json():
    s = make_settings()
    json_text = '{"candidate": "keep me intact"}'
    source = "x" * 50000
    out, info = fit_review_content("sys", source, json_text, s)
    assert info.truncated is True
    # JSON is protected in full; source is trimmed.
    assert json_text in out
    assert out.count("x") < 50000


def test_review_no_room_for_source_raises():
    # A huge JSON leaves no budget for any source text.
    s = make_settings(context_window=2048, max_tokens=1024, context_safety_margin=256)
    big_json = '{"x": "' + ("y" * 20000) + '"}'
    with pytest.raises(RuntimeError, match="No room for source text"):
        fit_review_content("sys", "source", big_json, s)
