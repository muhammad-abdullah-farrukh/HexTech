"""Thin wrapper so all stages share one call_llm().

Talks to the same LLM as ocr_resume_parser: an OpenAI-compatible endpoint
(llama.cpp / llama-server on :8080/v1) serving deepseek-r1-32b, via
{LLM_BASE_URL}/chat/completions. deepseek-r1 emits its reasoning inside
<think>…</think> in the message content; that block is stripped here so stages
only ever see the real answer (they parse plain text / JSON, not instructor).
"""
import re
import requests
import sys
import time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_TEMPERATURE

MAX_RETRIES   = 5
RETRY_BACKOFF = 5    # seconds; doubles each retry: 5s, 10s, 20s
REQUEST_TIMEOUT = 300  # generous backstop; reasoning models are slow

# Matches a leading/complete <think>…</think> reasoning block (deepseek-r1).
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove deepseek-r1 <think>…</think> reasoning, leaving only the answer.

    Handles the common case (one closed block) and the truncated case (an
    unclosed <think> that ran until the token cap — everything is reasoning,
    so there's no real answer to keep)."""
    cleaned = _THINK_RE.sub("", text)
    # An unclosed <think> means the whole response was reasoning that got cut
    # off before any answer — drop it rather than leak reasoning downstream.
    if "<think>" in cleaned.lower() and "</think>" not in cleaned.lower():
        cleaned = cleaned[: cleaned.lower().index("<think>")]
    return cleaned.strip()


def call_llm(prompt: str, max_tokens: int | None = None,
             presence_penalty: float | None = None,
             frequency_penalty: float | None = None,
             temperature: float | None = None,
             guided_json: dict | None = None,
             return_finish_reason: bool = False) -> str | tuple[str, str]:
    """Single-turn call to the shared OpenAI-compatible LLM endpoint.

    POSTs to {LLM_BASE_URL}/chat/completions and returns the assistant message
    content with any <think> reasoning stripped.

    Retries on timeout / connection errors / 5xx with exponential backoff.
    Raises the last exception if all retries are exhausted.

    max_tokens: caps generated length (maps to the OpenAI `max_tokens` field).

    presence_penalty / frequency_penalty: passed through when provided
    (supported by the OpenAI-compatible API).

    guided_json: a vLLM-only structured-output hint. Not enforced here; the
    stages validate/repair JSON downstream. Ignored with a one-line warning so
    callers that still pass it don't change behaviour.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty

    if guided_json is not None:
        print("[llm] ⚠ guided_json was requested but is not enforced on this "
              "endpoint — ignoring. Validate/repair JSON output downstream if "
              "structure isn't guaranteed.", flush=True)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            content = _strip_think(choice["message"]["content"] or "")
            finish_reason = choice.get("finish_reason", "stop")
            if not content:
                print("[llm] ⚠ empty content after stripping reasoning — check "
                      "max_tokens isn't too tight for this prompt.", flush=True)
            if finish_reason == "length":
                print(f"[llm] ⚠ output truncated by max_tokens={max_tokens} "
                      f"— response was cut off mid-generation.", flush=True)
            return (content, finish_reason) if return_finish_reason else content

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"[llm] ✗ attempt {attempt}/{MAX_RETRIES} failed "
                      f"({type(e).__name__}) — retrying in {wait}s …", flush=True)
                time.sleep(wait)
            else:
                print(f"[llm] ✗ attempt {attempt}/{MAX_RETRIES} failed "
                      f"({type(e).__name__}) — giving up.", flush=True)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_exc = e
            if status and 500 <= status < 600 and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"[llm] ✗ attempt {attempt}/{MAX_RETRIES} failed "
                      f"(HTTP {status}) — retrying in {wait}s …", flush=True)
                time.sleep(wait)
            else:
                print(f"[llm] ✗ attempt {attempt}/{MAX_RETRIES} failed "
                      f"(HTTP {status}) — not retrying (client error or out of retries). "
                      f"Response body: {e.response.text[:500] if e.response is not None else 'N/A'}", flush=True)
                raise

    raise last_exc


def call_llm_answer(prompt: str, max_tokens: int, *,
                    retry_budget: int | None = None, **kw) -> tuple[str, bool]:
    """Call the LLM for a real (short) answer that follows a <think> block.

    deepseek-r1 spends its budget reasoning before answering; if max_tokens is
    too small the response is cut off mid-<think> and _strip_think returns "".
    Returns (content, truncated). `truncated` is True when the model was still
    reasoning at the cap (finish_reason == "length") or nothing survived the
    think-strip.

    IMPORTANT for callers: truncated=True means "no trustworthy answer" — an
    INDETERMINATE result to flag/skip, NOT a negative/reject. Feeding the empty
    string into a yes/no parser (which reads "" as "no") is exactly the silent
    failure this wrapper exists to prevent.

    On truncation, retries ONCE at `retry_budget` (only if it exceeds
    max_tokens) before giving up.
    """
    content, finish = call_llm(
        prompt, max_tokens=max_tokens, return_finish_reason=True, **kw
    )
    truncated = (finish == "length") or (not content)
    if truncated and retry_budget and retry_budget > max_tokens:
        print(f"[llm] ↻ retrying truncated call at max_tokens={retry_budget} "
              f"(was {max_tokens})", flush=True)
        content, finish = call_llm(
            prompt, max_tokens=retry_budget, return_finish_reason=True, **kw
        )
        truncated = (finish == "length") or (not content)
    return content, truncated
