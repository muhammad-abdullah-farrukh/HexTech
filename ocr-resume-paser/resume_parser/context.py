"""Context-window budgeting.

The local model has a small context window (e.g. 4096). Before an LLM call we must
ensure: system_prompt + user_text + generated(max_tokens) <= context_window.

`fit_user_content` keeps the system prompt and reserved generation budget intact and
truncates only the resume text if it would overflow. Token counts are estimated from
a chars-per-token ratio (no tokenizer dependency); the safety margin absorbs the
estimate's slack plus chat-template overhead. Tune `LLM_CHARS_PER_TOKEN` /
`LLM_CONTEXT_SAFETY_MARGIN` if your model's tokenizer runs denser.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetInfo:
    context_window: int
    reserved_for_output: int
    system_tokens: int
    user_tokens_before: int
    user_tokens_after: int
    available_for_user: int
    truncated: bool
    # For the review pass: tokens of the protected (never-truncated) JSON block.
    protected_tokens: int = 0


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Rough token estimate from character count (ceil)."""
    if not text:
        return 0
    return math.ceil(len(text) / max(chars_per_token, 1e-6))


def fit_user_content(
    system_prompt: str, user_text: str, settings: Settings
) -> tuple[str, BudgetInfo]:
    """Return user_text (truncated if needed) that fits the context window.

    Raises if there is no room for any user content (system prompt + reserved
    output already exceed the window) — that's a config problem to surface, not
    something to silently paper over.
    """
    cpt = settings.chars_per_token
    reserved = settings.max_tokens + settings.context_safety_margin
    sys_tokens = estimate_tokens(system_prompt, cpt)
    available = settings.context_window - reserved - sys_tokens

    if available <= 0:
        raise RuntimeError(
            "No room for resume text: system prompt "
            f"(~{sys_tokens} tok) + max_tokens ({settings.max_tokens}) + margin "
            f"({settings.context_safety_margin}) exceed context_window "
            f"({settings.context_window}). Lower LLM_MAX_TOKENS or raise "
            "LLM_CONTEXT_WINDOW."
        )

    user_tokens = estimate_tokens(user_text, cpt)
    if user_tokens <= available:
        info = BudgetInfo(
            context_window=settings.context_window,
            reserved_for_output=settings.max_tokens,
            system_tokens=sys_tokens,
            user_tokens_before=user_tokens,
            user_tokens_after=user_tokens,
            available_for_user=available,
            truncated=False,
        )
        return user_text, info

    max_chars = int(available * cpt)
    truncated_text = user_text[:max_chars]
    logger.warning(
        "Resume text ~%d tok exceeds the ~%d tok budget for this %d-token context "
        "window; truncating to fit (kept %d of %d chars). Raise LLM_CONTEXT_WINDOW "
        "if your server allows a larger -c.",
        user_tokens,
        available,
        settings.context_window,
        len(truncated_text),
        len(user_text),
    )
    info = BudgetInfo(
        context_window=settings.context_window,
        reserved_for_output=settings.max_tokens,
        system_tokens=sys_tokens,
        user_tokens_before=user_tokens,
        user_tokens_after=estimate_tokens(truncated_text, cpt),
        available_for_user=available,
        truncated=True,
    )
    return truncated_text, info


# Labels for the review-pass user message. Kept short; their tokens are counted.
_REVIEW_SOURCE_LABEL = "SOURCE RESUME TEXT:\n"
_REVIEW_JSON_LABEL = "\n\nCURRENT EXTRACTED JSON:\n"


def fit_review_content(
    system_prompt: str, source_text: str, json_text: str, settings: Settings
) -> tuple[str, BudgetInfo]:
    """Compose the review user message, truncating ONLY the source text.

    The verify/refine pass needs both the source and the current JSON. The JSON is
    protected (never truncated); if the source would overflow the window, only the
    source is trimmed. Raises if there is no room for any source text (system +
    JSON + reserved output already exceed the window).
    """
    cpt = settings.chars_per_token
    reserved = settings.max_tokens + settings.context_safety_margin
    sys_tokens = estimate_tokens(system_prompt, cpt)
    json_tokens = estimate_tokens(json_text, cpt)
    label_tokens = estimate_tokens(_REVIEW_SOURCE_LABEL + _REVIEW_JSON_LABEL, cpt)
    available = settings.context_window - reserved - sys_tokens - json_tokens - label_tokens

    if available <= 0:
        raise RuntimeError(
            "No room for source text in the review pass: system "
            f"(~{sys_tokens} tok) + current JSON (~{json_tokens} tok) + max_tokens "
            f"({settings.max_tokens}) + margin ({settings.context_safety_margin}) "
            f"exceed context_window ({settings.context_window}). Lower "
            "LLM_MAX_TOKENS/LLM_REFINE_PASSES or raise LLM_CONTEXT_WINDOW."
        )

    source_tokens = estimate_tokens(source_text, cpt)
    kept_source = source_text
    truncated = False
    if source_tokens > available:
        kept_source = source_text[: int(available * cpt)]
        truncated = True
        logger.warning(
            "Review pass: source ~%d tok exceeds the ~%d tok budget (ctx %d, "
            "JSON ~%d tok protected); truncating source to %d of %d chars.",
            source_tokens,
            available,
            settings.context_window,
            json_tokens,
            len(kept_source),
            len(source_text),
        )

    user_text = (
        f"{_REVIEW_SOURCE_LABEL}{kept_source}{_REVIEW_JSON_LABEL}{json_text}"
    )
    info = BudgetInfo(
        context_window=settings.context_window,
        reserved_for_output=settings.max_tokens,
        system_tokens=sys_tokens,
        user_tokens_before=source_tokens,
        user_tokens_after=estimate_tokens(kept_source, cpt),
        available_for_user=available,
        truncated=truncated,
        protected_tokens=json_tokens,
    )
    return user_text, info
