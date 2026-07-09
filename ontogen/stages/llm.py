"""Thin wrapper so all stages share one call_llm().

Talks to an OpenAI-compatible LLM endpoint via {LLM_BASE_URL}/chat/completions.
Reasoning models emit their reasoning inside <think>…</think> in the message
content; that block is stripped here so stages only ever see the real answer
(they parse plain text / JSON, not instructor).

Every attempt (success or failure) is appended to
outputs/logs/llm_calls.jsonl, recording elapsed time AND the model the
SERVER actually reports having used (the response body's own "model" field)
alongside the model this process asked for (config.LLM_MODEL) — see
scripts/summarize_llm_calls.py to aggregate this after a run.

Why this matters: LLM_BASE_URL can point at a load-balanced/multi-backend
endpoint that silently answers a request with a different model than the one
requested, with zero client-visible signal besides that "model" field.
Observed live 2026-07-09: a cluster endpoint answered ~80% of requests as
vLLM Qwen3.5-9B and ~20% as a different llama.cpp-served model (different
weights, different max context) — same URL, no error, no header difference.
Any per-stage log that only recorded the *requested* model name (the old
behaviour here) is blind to this; a pipeline timing/quality measurement is
only trustworthy once segmented by the model the server actually reports,
not by what config.py asked for.
"""
import json
import os
import re
import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from queue import Queue, Empty
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import (
    LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_TEMPERATURE, OUTPUTS_DIR,
    LLM_ENDPOINTS,
)

MAX_RETRIES   = 5
RETRY_BACKOFF = 5    # seconds; doubles each retry: 5s, 10s, 20s
REQUEST_TIMEOUT = 300  # generous backstop; reasoning models are slow

# The default endpoint for a bare call_llm() with no override — the primary
# (config.LLM_ENDPOINTS[0]). A dict: {"base_url", "model", "slots", "ctx"}.
_PRIMARY_ENDPOINT = LLM_ENDPOINTS[0]

# Every call_llm() attempt across every stage/résumé lands here — the single
# choke point all stages share, so this is the one place instrumentation
# needs to live rather than being re-added per call site (and forgotten at
# some of them, as the old per-stage "model" log fields were).
_CALL_LOG_PATH = OUTPUTS_DIR / "logs" / "llm_calls.jsonl"

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


def _log_call(entry: dict) -> None:
    """Append one attempt (success or failure) to the shared call log.
    Best-effort: a logging failure must never break an actual LLM call."""
    try:
        _CALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CALL_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _base_log_entry(outcome: str, attempt: int, elapsed: float, prompt: str,
                     max_tokens: int | None,
                     requested_model: str, base_url: str) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outcome": outcome,
        "attempt": attempt,
        "elapsed_seconds": round(elapsed, 2),
        "requested_model": requested_model,
        "base_url": base_url,
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
    }


def call_llm(prompt: str, max_tokens: int | None = None,
             presence_penalty: float | None = None,
             frequency_penalty: float | None = None,
             temperature: float | None = None,
             guided_json: dict | None = None,
             return_finish_reason: bool = False,
             endpoint: dict | None = None) -> str | tuple[str, str]:
    """Single-turn call to an OpenAI-compatible LLM endpoint.

    POSTs to {endpoint base_url}/chat/completions and returns the assistant
    message content with any <think> reasoning stripped.

    Retries on timeout / connection errors / 5xx with exponential backoff.
    Raises the last exception if all retries are exhausted.

    max_tokens: caps generated length (maps to the OpenAI `max_tokens` field).

    presence_penalty / frequency_penalty: passed through when provided
    (supported by the OpenAI-compatible API).

    guided_json: a vLLM-only structured-output hint. Not enforced here; the
    stages validate/repair JSON downstream. Ignored with a one-line warning so
    callers that still pass it don't change behaviour.

    endpoint: which server to hit — a dict {"base_url", "model", ...} from
    config.LLM_ENDPOINTS. Defaults to the primary (index 0). call_llm_parallel
    passes each worker's assigned endpoint here; single callers omit it and get
    the primary, so their behaviour is unchanged.
    """
    ep = endpoint or _PRIMARY_ENDPOINT
    base_url = ep["base_url"]
    model    = ep["model"]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # Disable the reasoning trace. Both live models are reasoning models whose
    # thinking (a) can slip past _strip_think and (b) blows max_tokens, truncating
    # the real answer — the Stage 6 / EDC / Stage 9-10 failures of 2026-07-09.
    # A/B'd the same day across every pipeline task (verify yes/no, EDC define,
    # Stage 9-10 JSON extraction): suppressing thinking gave identical-or-better
    # output (fixed two wrong verdicts, cleaner JSON typing) with no dropped
    # fields / malformed JSON / hallucinated triples, at 2-100x the speed.
    #
    # The two backends need DIFFERENT suppression, so it's per-endpoint
    # (config: "think_suppress"):
    #   "flag"    — cluster vLLM/Qwen: honours chat_template_kwargs enable_thinking.
    #   "prefill" — local llama-server/DeepSeek-R1: has NO enable_thinking switch,
    #               so prefill an empty <think></think> and continue from it; the
    #               echoed block is removed downstream by _strip_think.
    # Escape hatch: HEXTECH_LLM_ENABLE_THINKING=1 restores the trace everywhere.
    suppress = ep.get("think_suppress")
    if suppress and os.environ.get("HEXTECH_LLM_ENABLE_THINKING", "").strip() not in ("1", "true", "True"):
        if suppress == "flag":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif suppress == "prefill":
            payload["messages"].append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False

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
        t0 = time.time()
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0

            # The response body's own "model" field is the server's word on
            # what actually answered — the only way to detect an endpoint
            # silently routing this specific call to a different backend than
            # the one requested. Absent on some servers; treat missing as
            # "unknown", not as a match.
            served_model = data.get("model") or ""
            mismatch = bool(served_model) and served_model != model

            choice = data["choices"][0]
            content = _strip_think(choice["message"]["content"] or "")
            if not content:
                print("[llm] ⚠ empty content after stripping reasoning — check "
                      "max_tokens isn't too tight for this prompt.", flush=True)
            finish_reason = choice.get("finish_reason", "stop")
            if finish_reason == "length":
                print(f"[llm] ⚠ output truncated by max_tokens={max_tokens} "
                      f"— response was cut off mid-generation.", flush=True)

            usage = data.get("usage") or {}
            log_entry = _base_log_entry("success", attempt, elapsed, prompt,
                                        max_tokens, model, base_url)
            log_entry.update({
                "served_model": served_model,
                "model_mismatch": mismatch,
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": finish_reason,
            })
            _log_call(log_entry)

            if mismatch:
                print(
                    f"[llm] ⚠ MODEL MISMATCH — requested {model!r} at {base_url} "
                    f"but the server answered as {served_model!r} ({elapsed:.1f}s). "
                    f"This endpoint is routing to more than one backend — this "
                    f"call's timing/output can't be attributed to the requested "
                    f"model. See outputs/logs/llm_calls.jsonl "
                    f"(scripts/summarize_llm_calls.py to aggregate).",
                    flush=True,
                )

            return (content, finish_reason) if return_finish_reason else content

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            elapsed = time.time() - t0
            last_exc = e
            _log_call(_base_log_entry(type(e).__name__, attempt, elapsed, prompt,
                                      max_tokens, model, base_url))
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"[llm] ✗ attempt {attempt}/{MAX_RETRIES} failed "
                      f"({type(e).__name__}) — retrying in {wait}s …", flush=True)
                time.sleep(wait)
            else:
                print(f"[llm] ✗ attempt {attempt}/{MAX_RETRIES} failed "
                      f"({type(e).__name__}) — giving up.", flush=True)

        except requests.exceptions.HTTPError as e:
            elapsed = time.time() - t0
            status = e.response.status_code if e.response is not None else None
            last_exc = e
            _log_call(_base_log_entry(f"http_{status}", attempt, elapsed, prompt,
                                      max_tokens, model, base_url))
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


# ── Parallel fan-out across endpoints ────────────────────────────────────────

class _MISSING:
    """Sentinel for a result slot no worker has filled yet."""


def call_llm_parallel(
    tasks: list[dict],
    endpoints: list[dict] | None = None,
    on_error: str = "raise",
    min_ctx: int | None = None,
) -> list:
    """Run many INDEPENDENT call_llm() invocations concurrently across the
    configured endpoints, preserving input order.

    tasks: one dict per call, each holding call_llm()'s keyword args, e.g.
        {"prompt": "...", "max_tokens": 1000}
        {"prompt": "...", "return_finish_reason": True, "temperature": 0.1}
    The "endpoint" key must NOT be set here — it is assigned by the dispatcher.

    Concurrency model: for each endpoint, `slots` worker threads are spawned
    (config default: local -np 3 + cluster 3 = 6 concurrent). All workers pull
    from one shared queue, so a faster endpoint naturally drains more tasks —
    no static pre-assignment that would leave fast workers idle behind a slow
    one. Each call still gets call_llm()'s own 5-try backoff; on top of that, a
    task whose assigned endpoint fails outright fails over to the OTHER
    endpoints once each before giving up, so one endpoint going down degrades
    to single-endpoint throughput instead of failing half the tasks.

    Returns a list of results in the SAME order as `tasks` (each is whatever
    call_llm returned — a str, or a (content, finish_reason) tuple).

    on_error:
      "raise"  (default) — after all tasks settle, re-raise the first
                exception, matching the fail-fast behaviour of the sequential
                loops this replaces.
      "return" — put the Exception object in that task's result slot instead;
                the caller inspects and decides. Used where a single bad item
                shouldn't sink the whole batch.

    min_ctx: if given, only dispatch to endpoints whose declared "ctx" is at
      least this (in tokens) — for a batch whose prompts are too large for a
      small-context endpoint (e.g. Stage 2 sends the full résumé; the local
      server's per-request window is 8192 vs the cluster's 32768). Endpoints
      without a "ctx"
      key are assumed to fit. If the filter removes every endpoint, it's
      ignored (all endpoints kept) with a warning — better to try and maybe
      truncate than to run nothing.
    """
    eps = endpoints if endpoints is not None else LLM_ENDPOINTS
    if min_ctx is not None:
        fitting = [e for e in eps if e.get("ctx", min_ctx) >= min_ctx]
        if fitting:
            eps = fitting
        else:
            print(
                f"[llm] ⚠ no endpoint's ctx >= {min_ctx} tokens — dispatching to "
                f"all anyway; prompts may be truncated/rejected.", flush=True,
            )
    n = len(tasks)
    results: list = [_MISSING] * n

    if n == 0:
        return results

    # Single endpoint, single slot → just run sequentially (keeps tests and
    # the HEXTECH_LLM_SINGLE_ENDPOINT escape hatch simple and deterministic).
    q: Queue = Queue()
    for i in range(n):
        q.put(i)

    def _worker(assigned_ep: dict) -> None:
        # Failover order: this worker's endpoint first, then the others once
        # each. call_llm already retried the assigned one 5x before raising.
        failover = [assigned_ep] + [e for e in eps if e is not assigned_ep]
        while True:
            try:
                idx = q.get_nowait()
            except Empty:
                return
            task = tasks[idx]
            call_kwargs = {k: v for k, v in task.items() if k != "endpoint"}
            last_exc = None
            for candidate in failover:
                try:
                    results[idx] = call_llm(endpoint=candidate, **call_kwargs)
                    last_exc = None
                    break
                except Exception as e:  # noqa: BLE001 — recorded / re-raised below
                    last_exc = e
            if last_exc is not None:
                results[idx] = last_exc

    # Build the worker set: `slots` threads per endpoint, each bound to it.
    worker_endpoints: list[dict] = []
    for ep in eps:
        worker_endpoints.extend([ep] * max(1, int(ep.get("slots", 1))))

    with ThreadPoolExecutor(max_workers=len(worker_endpoints)) as pool:
        for ep in worker_endpoints:
            pool.submit(_worker, ep)

    if on_error == "raise":
        for r in results:
            if isinstance(r, BaseException):
                raise r
    return results
