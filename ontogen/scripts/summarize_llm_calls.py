"""
Summarize outputs/logs/llm_calls.jsonl — the per-attempt call log every
stages/llm.py:call_llm() invocation writes (see that module's docstring for
why: LLM_BASE_URL can silently route a request to a different backend than
config.LLM_MODEL names, and this is the only place that's detectable).

The log is global and cumulative across every pipeline run, so use --since /
--until (or --run, which reads a résumé's own pipeline_<uuid>.log to find its
"PIPELINE: run started"/"run finished" timestamps automatically) to scope a
summary to one run instead of the whole file's history.

Run:
    python scripts/summarize_llm_calls.py
    python scripts/summarize_llm_calls.py --run 62103ed0-dbba-42f0-90ad-4c80c07a89bf
    python scripts/summarize_llm_calls.py --since 2026-07-09T08:00:00 --until 2026-07-09T09:30:00
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUTS_DIR

_CALL_LOG_PATH = OUTPUTS_DIR / "logs" / "llm_calls.jsonl"


def _run_window(resume_id: str) -> tuple[str | None, str | None]:
    """Read outputs/logs/pipeline_<uuid>.log and return the timestamps of the
    LAST 'PIPELINE: run started' / 'run finished' pair — i.e. the most recent
    run of this résumé, which is what you want right after kicking one off."""
    log_path = OUTPUTS_DIR / "logs" / f"pipeline_{resume_id}.log"
    if not log_path.exists():
        print(f"[summarize] ✗ no such log: {log_path}", file=sys.stderr)
        return None, None
    started = finished = None
    for line in log_path.read_text().splitlines():
        m = re.match(r"^(\S+) PIPELINE: run started", line)
        if m:
            started, finished = m.group(1), None
            continue
        m = re.match(r"^(\S+) PIPELINE: run finished", line)
        if m:
            finished = m.group(1)
    return started, finished


def summarize(since: str | None, until: str | None) -> None:
    if not _CALL_LOG_PATH.exists():
        print(f"[summarize] no call log at {_CALL_LOG_PATH} yet.")
        return

    total = 0
    in_window = 0
    by_served: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "seconds": 0.0, "mismatches": 0}
    )
    by_outcome: dict[str, int] = defaultdict(int)
    mismatches = 0

    with open(_CALL_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            ts = entry.get("timestamp", "")
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            in_window += 1

            outcome = entry.get("outcome", "unknown")
            by_outcome[outcome] += 1

            served = entry.get("served_model") or "(unknown — request failed or field absent)"
            bucket = by_served[served]
            bucket["count"] += 1
            bucket["seconds"] += entry.get("elapsed_seconds", 0.0)
            if entry.get("model_mismatch"):
                bucket["mismatches"] += 1
                mismatches += 1

    if since or until:
        print(f"Window: {since or '(start)'} → {until or '(end)'}")
    print(f"{in_window}/{total} logged attempts in window\n")

    print("By outcome:")
    for outcome, count in sorted(by_outcome.items(), key=lambda kv: -kv[1]):
        print(f"  {outcome:20s} {count}")

    print("\nBy served model (what the server actually reported answering with):")
    print(f"  {'model':45s} {'calls':>6s} {'total s':>10s} {'avg s':>8s} {'mismatches':>11s}")
    for served, b in sorted(by_served.items(), key=lambda kv: -kv[1]["seconds"]):
        avg = b["seconds"] / b["count"] if b["count"] else 0.0
        print(f"  {served:45s} {b['count']:6d} {b['seconds']:10.1f} {avg:8.1f} {b['mismatches']:11d}")

    if mismatches:
        print(
            f"\n⚠ {mismatches} call(s) were answered by a DIFFERENT model than "
            f"requested — this endpoint routed to more than one backend during "
            f"this window. Any single 'total pipeline time' figure over this "
            f"window is not attributable to one model."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", help="ISO timestamp lower bound (inclusive)")
    parser.add_argument("--until", help="ISO timestamp upper bound (inclusive)")
    parser.add_argument(
        "--run", metavar="RESUME_UUID",
        help="scope to this résumé's most recent pipeline run window "
             "(reads outputs/logs/pipeline_<uuid>.log)",
    )
    args = parser.parse_args()

    since, until = args.since, args.until
    if args.run:
        run_started, run_finished = _run_window(args.run)
        if run_started is None:
            return
        since = since or run_started
        until = until or run_finished
        if run_finished is None:
            print("[summarize] ⚠ no 'run finished' line yet — run may still be in progress; "
                  "summarizing everything from run start to now.\n")

    summarize(since, until)


if __name__ == "__main__":
    main()
