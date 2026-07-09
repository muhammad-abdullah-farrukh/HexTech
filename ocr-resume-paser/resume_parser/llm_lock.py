"""Shared, advisory LLM lock for the parser ↔ Ontogen sequence.

Both the parser and Ontogen call one llama-server running ``--parallel 1``, so
only one of them may drive it at a time. This module provides a fail-safe file
lock plus a reachability pre-flight, used symmetrically by both sides
(resume_parser.cli and ontogen/pipeline.py). It lives in the parser package
because Ontogen already imports ``resume_parser`` — the dependency only ever
points that way.

The lock is *advisory*: it protects against two runs colliding on the single
inference slot, not against arbitrary misuse. A lock left behind by a dead
process is detected (PID no longer alive) and cleaned up automatically, so a
crash never requires manual file deletion.
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests

# One fixed, location-independent path so the parser and Ontogen agree on it
# regardless of which working directory each is launched from.
LOCK_PATH = Path(tempfile.gettempdir()) / "hextech_llm.lock"
REACHABILITY_TIMEOUT = 5  # seconds


def _pid_alive(pid: int) -> bool:
    """True if a process with `pid` currently exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def _check_reachable(base_url: str) -> None:
    """Confirm the LLM endpoint answers before we start a long run."""
    url = f"{base_url.rstrip('/')}/models"
    try:
        requests.get(url, timeout=REACHABILITY_TIMEOUT)
    except requests.RequestException as exc:
        raise SystemExit(
            f"[llm-lock] LLM endpoint not reachable at {url} ({type(exc).__name__}: {exc}). "
            f"Is the llama-server running? Aborting before doing any work."
        )


def _read_lock() -> dict | None:
    try:
        return json.loads(LOCK_PATH.read_text())
    except (FileNotFoundError, ValueError):
        return None


def _acquire() -> None:
    existing = _read_lock()
    if existing is not None:
        pid = existing.get("pid")
        if isinstance(pid, int) and pid != os.getpid() and _pid_alive(pid):
            raise SystemExit(
                f"[llm-lock] LLM appears busy — parser or another Ontogen run may be "
                f"in progress [pid {pid}, started {existing.get('started_at')}]. "
                f"Aborting; try again once it finishes."
            )
        # Stale lock (owner is gone) — reclaim it.
        print(f"[llm-lock] clearing stale lock from pid {pid}", flush=True)

    LOCK_PATH.write_text(
        json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()})
    )


def _release() -> None:
    current = _read_lock()
    # Only remove the lock if it is still ours (don't stomp a reclaimer's lock).
    if current is not None and current.get("pid") == os.getpid():
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def llm_lock(base_url: str):
    """Guard an LLM-driven run: pre-flight the endpoint, take the advisory lock,
    and always release it on exit."""
    _check_reachable(base_url)
    _acquire()
    try:
        yield
    finally:
        _release()
