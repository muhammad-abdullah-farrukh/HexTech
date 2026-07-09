"""Runtime configuration loaded from the environment / a .env file.

The pipeline talks to a single local, OpenAI-compatible LLM endpoint (e.g. a
vLLM server). `load_settings()` is the single entry point; it reads the env once
and returns a frozen `Settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_INSTRUCTOR_MODE = "JSON_SCHEMA"

# Order the probe (and the runtime fallback ladder) walk down. The configured
# mode is tried first, then the remaining rungs below it in this order.
MODE_LADDER = ["JSON_SCHEMA", "TOOLS", "JSON", "MD_JSON"]


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str
    instructor_mode: str
    max_retries: int
    ratelimit_attempts: int
    health_timeout: float
    # Self-verify/refine passes after the initial generate. Total LLM calls per
    # resume = 1 + refine_passes. 0 disables the loop.
    refine_passes: int
    # Context-window budgeting. The cleaned resume is truncated if
    # system_prompt + resume + max_tokens would exceed context_window.
    context_window: int
    context_safety_margin: int
    chars_per_token: float
    # Sampling guards (applied to every LLM call). Anti-repetition penalties + a
    # token cap keep degenerate/small models from runaway repetition loops.
    temperature: float
    frequency_penalty: float
    presence_penalty: float
    max_tokens: int


def load_settings(env_path: str | os.PathLike[str] | None = None) -> Settings:
    """Load and validate settings.

    Reads `env_path` (or the default .env discovery) into os.environ, then builds
    a `Settings`. Raises if the model name is missing or the mode is unknown.
    """
    if env_path is not None:
        load_dotenv(Path(env_path), override=False)
    else:
        load_dotenv(override=False)

    model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "LLM_MODEL is not set. Copy .env.example to .env and set LLM_MODEL to "
            "the model name your local server serves (and pass --env if needed)."
        )

    mode = os.environ.get("INSTRUCTOR_MODE", DEFAULT_INSTRUCTOR_MODE).strip().upper()
    if mode not in MODE_LADDER:
        raise RuntimeError(f"INSTRUCTOR_MODE={mode!r} is not one of {MODE_LADDER}.")

    return Settings(
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip(),
        model=model,
        api_key=os.environ.get("LLM_API_KEY", "not-needed").strip(),
        instructor_mode=mode,
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", "2")),
        ratelimit_attempts=int(os.environ.get("LLM_RATELIMIT_ATTEMPTS", "5")),
        health_timeout=float(os.environ.get("LLM_HEALTH_TIMEOUT", "2.0")),
        refine_passes=int(os.environ.get("LLM_REFINE_PASSES", "2")),
        context_window=int(os.environ.get("LLM_CONTEXT_WINDOW", "4096")),
        context_safety_margin=int(os.environ.get("LLM_CONTEXT_SAFETY_MARGIN", "256")),
        chars_per_token=float(os.environ.get("LLM_CHARS_PER_TOKEN", "3.5")),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.6")),
        frequency_penalty=float(os.environ.get("LLM_FREQUENCY_PENALTY", "0.15")),
        presence_penalty=float(os.environ.get("LLM_PRESENCE_PENALTY", "0.0")),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
    )


def fallback_modes(start_mode: str) -> list[str]:
    """The mode ladder starting at `start_mode`, then the remaining rungs below it."""
    start = start_mode.upper()
    if start not in MODE_LADDER:
        return [start]
    idx = MODE_LADDER.index(start)
    return MODE_LADDER[idx:]
