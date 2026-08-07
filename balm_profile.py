"""
balm_profile.py
===============
Zero-dependency call-count + wall-time profiler for the BALM/ColabDesign bridge.

Usage (see bottom of file for the exact wiring):

    from balm_profile import timed, report, reset

    @timed("host_penalty")
    def host_penalty(seq_idx_np):
        ...

    # after each trajectory:
    report(); reset()

WHY A DECORATOR AND NOT AN INLINE TIMER: the interesting quantity is how many
times the HOST function actually executes, and decorators compose so you can
instrument several layers (host_penalty / raw_score / predictor.score) and read
off where the time splits, without editing each body.
"""
from __future__ import annotations

import functools
import time
from collections import OrderedDict

# name -> {"n": calls, "s": seconds, "cached": short-circuited calls}
_STATS: "OrderedDict[str, dict]" = OrderedDict()

# Set True to insert torch.cuda.synchronize() around timed regions. Torch is
# async on GPU, so WITHOUT this the timings attribute cost to whichever call
# happens to block next, and the ESM pass will look deceptively cheap.
CUDA_SYNC = False


def _sync():
    if not CUDA_SYNC:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timed(name=None):
    """Decorator: count calls and accumulate wall time under `name`."""
    def deco(fn):
        label = name or getattr(fn, "__name__", repr(fn))
        _STATS.setdefault(label, {"n": 0, "s": 0.0})

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _sync()
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _sync()
                st = _STATS[label]
                st["n"] += 1
                st["s"] += time.perf_counter() - t0
        wrapper._balm_timed = label
        return wrapper
    return deco


def wrap_method(obj, attr, name=None):
    """Instrument a BOUND METHOD in place, with no edit to its source file.

    Works because Python resolves `obj.attr` at call time, so replacing the
    instance attribute redirects subsequent calls. Idempotent.
    """
    fn = getattr(obj, attr)
    if getattr(fn, "_balm_timed", None):
        return fn
    wrapped = timed(name or f"{type(obj).__name__}.{attr}")(fn)
    setattr(obj, attr, wrapped)
    return wrapped


def stats() -> dict:
    return {k: dict(v) for k, v in _STATS.items()}


def reset():
    for v in _STATS.values():
        v["n"] = 0
        v["s"] = 0.0


def report(prefix="[balm-prof]", expected=None, printer=print):
    """Print one line per instrumented layer.

    expected: optional int, your predicted call count (e.g. iters * mutations).
              Flags the 'fires in every stage' bug when n is a large multiple.
    """
    if not _STATS:
        printer(f"{prefix} no instrumented calls recorded")
        return stats()
    printer(f"{prefix} {'layer':<28} {'calls':>8} {'total_s':>10} {'ms/call':>10}")
    for label, st in _STATS.items():
        n, s = st["n"], st["s"]
        per = (s / n * 1000.0) if n else 0.0
        printer(f"{prefix} {label:<28} {n:>8} {s:>10.3f} {per:>10.2f}")
    if expected:
        n0 = next(iter(_STATS.values()))["n"]
        if n0 > expected * 1.5:
            printer(f"{prefix} WARNING: {n0} calls vs ~{expected} expected "
                    f"({n0/expected:.1f}x) -- loss is firing outside the discrete stage.")
    if not CUDA_SYNC:
        printer(f"{prefix} note: CUDA_SYNC=False, GPU time may be under-attributed.")
    return stats()


# --------------------------------------------------------------------------- #
# WIRING
# --------------------------------------------------------------------------- #
"""
1) host_penalty MUST be decorated at its definition site in
   balm_colabdesign_bridge.py:

       from balm_profile import timed

       @timed("host_penalty")
       def host_penalty(seq_idx_np):
           ...

   You cannot instrument it afterwards via `loss_fn.host_penalty = ...`:
   jax.pure_callback captured the original function object in a closure, so
   reassigning that attribute changes what your tests see, not what the
   callback calls.

2) The layers BELOW it need no source edits -- monkeypatch after construction:

       from balm_profile import wrap_method, report, reset
       wrap_method(balm_aux, "raw_score")          # counts cache misses+hits
       wrap_method(balm_aux.predictor, "score")    # counts real ESM passes

3) Per trajectory, in binder_hallucination:

       report(expected=20*6); reset()

READING THE RESULT
   host_penalty n >> expected     -> attachment scope; move to stage 4 only.
   host_penalty n ~= predictor.score n  -> the score cache is doing nothing.
   raw_score n >> predictor.score n     -> cache is working, ESM is not the cost.
   all n sane, total_s << slowdown -> cost is the pure_callback pipeline stall,
                                      not BALM compute.

Set CUDA_SYNC=True for a second run to get true GPU attribution; leave it False
for the call counts, since synchronizing perturbs the thing you're measuring.
"""
