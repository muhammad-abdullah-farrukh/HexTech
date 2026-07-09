"""Local LLM client factory + rate-limit/retry wrapper.

`make_client` builds an instructor-wrapped OpenAI client pointed at the local
endpoint. `completion_with_backoff` wraps a single instructor call with tenacity
backoff that retries only on transient errors (connection/timeout, 5xx, 429) and
lets everything else (validation, bad request) propagate.

Three error classes stay distinct (see normalize.py):
  * transient / rate-limit -> retried here with backoff
  * validation failure     -> handled by instructor's `max_retries` re-ask
  * unrecoverable          -> surfaces after attempts are spent
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import instructor
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from .settings import Settings

logger = logging.getLogger(__name__)

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def mode_from_str(name: str) -> instructor.Mode:
    """Map an INSTRUCTOR_MODE string to an instructor.Mode enum value."""
    try:
        return instructor.Mode[name.upper()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown instructor mode {name!r}; expected one of "
            f"{[m.name for m in instructor.Mode]}"
        ) from exc


def sampling_kwargs(settings: Settings) -> dict[str, Any]:
    """Per-request sampling guards applied to every call."""
    kw: dict[str, Any] = {
        "temperature": settings.temperature,
        "frequency_penalty": settings.frequency_penalty,
        "presence_penalty": settings.presence_penalty,
    }
    if settings.max_tokens:
        kw["max_tokens"] = settings.max_tokens
    return kw


def is_endpoint_active(settings: Settings) -> bool:
    """Return True if the local OpenAI-compatible server answers GET /models."""
    try:
        probe = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key or "not-needed",
            timeout=settings.health_timeout,
            max_retries=0,
        )
        probe.models.list()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
        logger.info("Local endpoint %s not reachable: %s", settings.base_url, exc)
        return False


def make_client(settings: Settings, mode: str | None = None):
    """Build an instructor-wrapped OpenAI client for the local endpoint."""
    base = OpenAI(base_url=settings.base_url, api_key=settings.api_key or "not-needed")
    return instructor.from_openai(
        base, mode=mode_from_str(mode or settings.instructor_mode)
    )


def completion_with_backoff(
    create_fn: Callable[..., Any], attempts: int, **kwargs: Any
) -> Any:
    """Call `create_fn(**kwargs)`, retrying transient errors with backoff."""

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=2, min=2, max=60) + wait_random(0, 2),
        stop=stop_after_attempt(max(1, attempts)),
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            "LLM call transient error (%s); retry %d/%d",
            rs.outcome.exception().__class__.__name__,
            rs.attempt_number,
            attempts,
        ),
    )
    def _run() -> Any:
        return create_fn(**kwargs)

    return _run()
