from resume_parser.normalize import run_refine_loop


def test_loop_stops_on_first_approval():
    calls = []

    def refine_fn(current):
        calls.append(current)
        # approve immediately, with a small correction
        return True, {**current, "phone": "123"}

    out = run_refine_loop({"name": "A"}, refine_fn, max_passes=2)
    assert out == {"name": "A", "phone": "123"}
    assert len(calls) == 1  # stopped after first approval


def test_loop_stops_on_convergence():
    def refine_fn(current):
        # never "approved", but returns the same dict -> converged
        return False, dict(current)

    calls = {"n": 0}

    def counting(current):
        calls["n"] += 1
        return refine_fn(current)

    out = run_refine_loop({"name": "A"}, counting, max_passes=3)
    assert out == {"name": "A"}
    assert calls["n"] == 1  # converged after one pass (no change)


def test_loop_respects_max_passes():
    calls = {"n": 0}

    def refine_fn(current):
        calls["n"] += 1
        # never approved, always changes -> would loop forever without the cap
        return False, {**current, "v": current.get("v", 0) + 1}

    out = run_refine_loop({"v": 0}, refine_fn, max_passes=2)
    assert calls["n"] == 2
    assert out == {"v": 2}


def test_loop_disabled_with_zero_passes():
    def refine_fn(current):  # pragma: no cover - must never be called
        raise AssertionError("refine_fn should not be called when max_passes=0")

    out = run_refine_loop({"name": "A"}, refine_fn, max_passes=0)
    assert out == {"name": "A"}
