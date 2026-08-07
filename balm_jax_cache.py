"""
balm_jax_cache.py
=================
Enable JAX's PERSISTENT compilation cache, and count XLA compiles per trajectory
so you can prove the cache is actually hitting.

WHY
---
ColabDesign builds a fresh af_model every trajectory, so the in-memory jit cache
(keyed on the Python function object) always misses. The persistent cache is keyed
on the HLO hash instead, so an identical graph from a *new* model object hits it.
With a fixed binder length the pre-Stage-4 graph and the Stage-4 graph are each
compiled once for the whole run rather than once per trajectory.

IMPORT ORDER MATTERS
--------------------
The cache must be configured BEFORE JAX compiles anything. Setting the env var in
your SLURM script is the robust way, because it is read at JAX init regardless of
Python import order:

    export JAX_COMPILATION_CACHE_DIR=/node-local/scratch/jax_cache

This module is a belt-and-braces fallback: import it before `import jax` and it
sets the same thing programmatically, tolerating the API rename across versions.
"""
from __future__ import annotations

import logging
import os

DEFAULT_MIN_COMPILE_SECS = 1.0   # don't cache trivial compiles
_STATE = {"configured": False, "dir": None, "method": None}
_COMPILES = {"n": 0}


# --------------------------------------------------------------------------- #
# Cache configuration                                                          #
# --------------------------------------------------------------------------- #
def configure(cache_dir=None, min_compile_secs=DEFAULT_MIN_COMPILE_SECS, verbose=True):
    """Point JAX at a persistent compilation cache. Idempotent.

    cache_dir: defaults to $JAX_COMPILATION_CACHE_DIR. Use NODE-LOCAL fast storage;
               a shared NFS path across array jobs trades compile time for lock
               contention.
    Returns the directory in use, or None if not configured.
    """
    if _STATE["configured"]:
        return _STATE["dir"]

    cache_dir = cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        if verbose:
            print("[jax-cache] no cache dir set (JAX_COMPILATION_CACHE_DIR unset); "
                  "AF2 will recompile every trajectory.")
        _STATE["configured"] = True
        return None

    try:
        os.makedirs(cache_dir, exist_ok=True)
        probe = os.path.join(cache_dir, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except Exception as e:
        print(f"[jax-cache] ERROR: cache dir {cache_dir!r} not writable ({e}); "
              f"caching DISABLED.")
        _STATE["configured"] = True
        return None

    # Env var is read at JAX init; set it too in case JAX has not started yet.
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir

    method = None
    try:
        import jax
        # Newer JAX (>= ~0.4.24): config flags.
        try:
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_secs)
            method = "jax.config"
        except Exception:
            # Older JAX: explicit initializer.
            from jax.experimental.compilation_cache import compilation_cache as cc
            cc.initialize_cache(cache_dir)
            method = "experimental.initialize_cache"
    except Exception as e:
        method = f"env-var-only ({e})"

    _STATE.update({"configured": True, "dir": cache_dir, "method": method})
    if verbose:
        print(f"[jax-cache] persistent compilation cache: {cache_dir} (via {method})")
    return cache_dir


# --------------------------------------------------------------------------- #
# Compile counting                                                             #
# --------------------------------------------------------------------------- #
class _CompileCountingHandler(logging.Handler):
    """Counts XLA compilation events from JAX's own log records.

    More reliable than eyeballing JAX_LOG_COMPILES in a SLURM log, and it gives a
    per-trajectory number you can assert on.
    """

    KEYS = ("Compiling ", "Finished XLA compilation", "Finished tracing + transforming")

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if any(k in msg for k in self.KEYS):
            _COMPILES["n"] += 1


def install_compile_counter(verbose=True):
    """Turn on JAX compile logging and attach the counter. Idempotent."""
    if _STATE.get("counter"):
        return
    try:
        import jax
        jax.config.update("jax_log_compiles", True)
    except Exception as e:
        if verbose:
            print(f"[jax-cache] could not enable jax_log_compiles: {e}")
    handler = _CompileCountingHandler(level=logging.DEBUG)
    for name in ("jax", "jax._src.dispatch", "jax._src.interpreters.pxla",
                 "jax._src.compiler"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)
    _STATE["counter"] = handler
    if verbose:
        print("[jax-cache] compile counter installed")


def compiles():
    return _COMPILES["n"]


def reset_compiles():
    _COMPILES["n"] = 0


def report_compiles(tag="", printer=print):
    """Print and reset the compile count for the window just finished.

    EXPECTED PATTERN with a working cache and a fixed binder length:
        trajectory 1 -> a handful of compiles (each graph, once)
        trajectory 2+ -> 0
    A nonzero count on every trajectory means the cache is NOT hitting: usually a
    shape that varies (binder length, num_recycles via optimise_beta) or a cache
    dir that is not writable.
    """
    n = _COMPILES["n"]
    printer(f"[jax-cache] XLA compiles during {tag or 'window'}: {n}")
    reset_compiles()
    return n


def status():
    return {"dir": _STATE["dir"], "method": _STATE["method"],
            "counter": bool(_STATE.get("counter")), "compiles": _COMPILES["n"]}