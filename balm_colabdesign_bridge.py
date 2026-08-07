"""
balm_colabdesign_bridge_profiled.py
===================================
Drop-in replacement for balm_colabdesign_bridge.py with the profiling decorator
wired in at the only place it works (the host side of the pure_callback), plus a
one-time device/dtype banner to catch silent CPU fallback.

Behaviour is IDENTICAL to the original when profiling is unavailable -- the
balm_profile import is optional and degrades to a no-op decorator.

WHAT CHANGED vs the original
  1. host_penalty is decorated with @timed("host_penalty") AT ITS DEFINITION
     SITE. This is the only correct place: jax.pure_callback closes over the
     function object, so post-hoc reassignment of loss_fn.host_penalty would
     instrument something the callback never calls.
  2. A one-time banner prints the torch device + dtype the ESM model actually
     ended up on, so a `cuda`-requested-but-CPU-loaded model is loud, not silent.
  3. active_flag short-circuit is unchanged, but now COUNTED separately, because
     it skips ESM while still paying the full pure_callback device stall -- the
     two costs need to be distinguishable in the report.
"""
from __future__ import annotations

# AlphaFold / ColabDesign canonical residue order.
# Matches "".join(residue_constants.restypes) -> "ARNDCQEGHILKMFPSTWYV"
RESTYPES = "ARNDCQEGHILKMFPSTWYV"

try:
    from balm_profile import timed
except Exception:  # profiling is optional; never break the design run for it
    def timed(name=None):
        def deco(fn):
            return fn
        return deco


def _indices_to_sequence(idx, alphabet=RESTYPES):
    """(L,) int array of restype indices -> amino-acid string."""
    return "".join(alphabet[int(i)] if 0 <= int(i) < len(alphabet) else "X" for i in idx)


def _banner_once(aux_loss, state={}):
    """Print the ESM model's real device/dtype exactly once. Cheap insurance
    against build_model()'s silent CPU fallback eating your wall clock."""
    if state.get("done"):
        return
    state["done"] = True
    try:
        model = aux_loss.predictor._model
        if model is None:
            return  # not built yet; banner will fire on a later call
        p = next(model.parameters())
        print(f"[balm] ESM live on device={p.device} dtype={p.dtype} "
              f"training={model.training}")
        import torch
        if p.device.type == "cpu":
            print("[balm] WARNING: ESM is on CPU. Expect ~1-2 s per score. "
                  "This alone can dominate trajectory time.")
        elif torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"[balm] torch sees {free/2**30:.1f} GiB free of "
                  f"{total/2**30:.1f} GiB (rest may be held by JAX preallocation)")
    except Exception as e:  # never let diagnostics kill a run
        print(f"[balm] banner skipped: {e}")


def add_balm_loss(self, aux_loss, weight, binder_len=None, alphabet=RESTYPES,
                  stop_grad=True, key="balm", active_flag=None, profile=True):
    """Append a BALM affinity loss to a ColabDesign af_model (BindCraft idiom).

    self       : af_model from mk_afdesign_model(protocol="binder").
    aux_loss   : BalmAuxLoss -- supplies .raw_score(binder) (cached) + target seq.
    weight     : float from JSON; registered as opt["weights"][key].
    binder_len : defaults to self._binder_len (set after prep_inputs()).
    stop_grad  : wrap output in stop_gradient (autodiff-safe).
    active_flag: optional {"on": bool}; when present and False, host scorer returns 0
                 without running ESM. NOTE: this skips the ESM pass but NOT the
                 pure_callback host round-trip, so it is cheaper, not free.
    profile    : count/time host-side calls via balm_profile (no-op if absent).
    Returns the registered loss_fn (also exposes .host_penalty for tests).
    """
    import numpy as np

    blen = int(binder_len) if binder_len is not None else int(getattr(self, "_binder_len"))

    _wrap = timed("host_penalty") if profile else (lambda f: f)
    _wrap_skip = timed("host_penalty(skipped)") if profile else (lambda f: f)

    @_wrap_skip
    def _inactive():
        # Counted separately: proves how many callbacks fire outside the
        # discrete stage, i.e. pure stall with no BALM work attached.
        return np.asarray(0.0, dtype=np.float32)

    @_wrap
    def host_penalty(seq_idx_np):
        if active_flag is not None and not active_flag.get("on", True):
            return _inactive()
        idx = np.asarray(seq_idx_np).reshape(-1)[-blen:]      # binder = last blen residues
        binder = _indices_to_sequence(idx, alphabet)
        cos = aux_loss.raw_score(binder)                      # cosine in [-1,1], cached
        _banner_once(aux_loss)
        # ColabDesign MINIMIZES loss. direction="maximize" -> penalty=-cos (drives cos up);
        # direction="minimize" -> penalty=+cos (drives cos down / more negative = better).
        sign = -1.0 if aux_loss.cfg.direction == "maximize" else 1.0
        return np.asarray(sign * float(cos), dtype=np.float32)

    def loss_fn(inputs, outputs):
        # NOTE: do NOT decorate loss_fn -- it runs under jit, so timing it would
        # measure tracing (once), not execution.
        import jax
        import jax.numpy as jnp
        seq = inputs["seq"]
        if isinstance(seq, dict):
            seq = seq.get("hard", seq.get("pseudo", next(iter(seq.values()))))
        seq = jnp.asarray(seq)
        if seq.ndim == 3:          # (num_seq, L, 20) -> first sequence
            seq = seq[0]
        seq_idx = jnp.argmax(seq, axis=-1)                    # (L,)
        penalty = jax.pure_callback(
            host_penalty,
            jax.ShapeDtypeStruct((), jnp.float32),
            seq_idx,
            vmap_method="sequential",   # remove this kwarg on jax < 0.4.24
        )
        if stop_grad:
            penalty = jax.lax.stop_gradient(penalty)
        return {key: penalty}

    loss_fn.host_penalty = host_penalty   # exposed for testing
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"][key] = weight
    return loss_fn


def remove_balm_loss(self, loss_fn, key="balm"):
    """Detach a previously added BALM loss. Use this to keep the callback out of
    the gradient stages entirely instead of gating it with active_flag -- gating
    still pays the pure_callback stall on every model call, detaching does not."""
    try:
        self._callbacks["model"]["loss"].remove(loss_fn)
    except ValueError:
        pass
    self.opt["weights"].pop(key, None)
