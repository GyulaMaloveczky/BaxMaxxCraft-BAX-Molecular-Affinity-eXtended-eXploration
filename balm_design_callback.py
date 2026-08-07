"""
balm_design_callback.py
=======================
BALM affinity steering for ColabDesign's DISCRETE stages, WITHOUT putting BALM
inside the jitted AF2 graph.

WHY THIS EXISTS (replaces balm_colabdesign_bridge.add_balm_loss)
----------------------------------------------------------------
The bridge followed BindCraft's `add_rg_loss` idiom: append to
`_callbacks["model"]["loss"]`. Those callbacks are TRACED into the jitted model
function, so an external PyTorch scorer had to be reached via
`jax.pure_callback`. Measured cost of that on a real run:

    Stage 4 without BALM+walls : ~69 s
    Stage 4 with    BALM+walls : ~292 s
    BALM's own compute         : 1.3 s

i.e. ~0.55 s of wall time per callback for ~3.3 ms of actual work. The callback
forces XLA to break the graph and drain the device on every invocation; it also
made `jit(_model)` compilation ~6x slower (9.7 s vs 1.6 s per compile).

None of that is necessary. Semigreedy/MCMC accept-reject on a SCALAR computed in
ordinary Python:

    design.py:465   losses = [x["aux"]["loss"] for x in buff]
    design.py:466   best   = buff[np.argmin(losses)]

and `run()` exposes a Python hook that fires after aux is built:

    design.py:109   for fn in (self._callbacks["design"]["post"] + ...): fn(self)
    design.py:112   self.aux["log"] = {**self.aux["losses"]}
    design.py:114   for k in ["loss","i_ptm","ptm"]: self.aux["log"][k] = self.aux[k]

So we mutate `self.aux["loss"]` at line 109 and line 114 propagates it into
`aux["log"]["loss"]` -- which is what `_save_results`/`save_best` and
`_design_mcmc` (design.py:546) read. Same steering, no graph change.

BONUS: the post hook fires ONCE PER `run()`, i.e. once per candidate, whereas the
traced loss fired once per model per recycle. On the measured config that is 404
BALM invocations -> ~100.

DELIBERATELY NOT TOUCHING opt["weights"]
----------------------------------------
`self.opt` is passed into the jitted model fn as a pytree. Adding a key to
opt["weights"] changes its treedef and would itself force a recompile -- exactly
what we are removing. The BALM weight is therefore held in this module, not in
opt. Side effect: `_print_log` filters printed terms by opt["weights"], so "balm"
will not appear in ColabDesign's per-iteration stdout line. It IS written into
aux["losses"], so it still reaches aux["log"], `_tmp["log"]`, and therefore
`af_model.get_loss("balm")` and your trajectory plots.
"""
from __future__ import annotations

RESTYPES = "ARNDCQEGHILKMFPSTWYV"

try:
    from balm_profile import timed
except Exception:
    def timed(name=None):
        def deco(fn):
            return fn
        return deco


def _indices_to_sequence(idx, alphabet=RESTYPES):
    """(L,) int array of restype indices -> amino-acid string."""
    return "".join(alphabet[int(i)] if 0 <= int(i) < len(alphabet) else "X" for i in idx)


def _extract_binder_seq(af, blen, alphabet=RESTYPES):
    """Pull the current binder sequence out of af.aux.

    aux["seq"] carries hard/pseudo/logits; hard is the one-hot used by the
    discrete stages. It may be (num_seq, L, 20) -- averaged over models by
    run()'s avg_or_first, which is harmless because every model saw the SAME
    sequence, so the argmax is unchanged.
    """
    import numpy as np
    seq = af.aux["seq"]
    if isinstance(seq, dict):
        seq = seq.get("hard", seq.get("pseudo", seq.get("logits")))
    seq = np.asarray(seq)
    if seq.ndim == 3:
        seq = seq[0]
    idx = seq.argmax(-1).reshape(-1)[-blen:]     # binder = last blen residues
    return _indices_to_sequence(idx, alphabet)


def add_balm_design_callback(af, aux_loss, weight, binder_len=None,
                             alphabet=RESTYPES, key="balm", active_flag=None,
                             verbose_first=True):
    """Steer a discrete stage with BALM via ColabDesign's Python post-hook.

    af         : af_model from mk_afdesign_model(protocol="binder").
    aux_loss   : BalmAuxLoss -- supplies .raw_score(binder) (cached) + direction.
    weight     : float; multiplies the penalty added to aux["loss"].
    binder_len : defaults to af._binder_len (set after prep_inputs()).
    active_flag: optional {"on": bool}; when False the hook returns immediately.
                 Unlike the traced version this really IS free -- no device sync.
    Returns the registered callback (detach with remove_balm_design_callback).
    """
    blen = int(binder_len) if binder_len is not None else int(getattr(af, "_binder_len"))
    state = {"n": 0}

    @timed("balm_post_callback")
    def post_callback(self):
        if active_flag is not None and not active_flag.get("on", True):
            return
        binder = _extract_binder_seq(self, blen, alphabet)
        cos = aux_loss.raw_score(binder)                 # cached; PyTorch, off-graph
        # ColabDesign MINIMIZES. direction="maximize" -> penalty=-cos (drives cos up);
        # direction="minimize" -> penalty=+cos. Identical convention to the old bridge.
        sign = -1.0 if aux_loss.cfg.direction == "maximize" else 1.0
        penalty = sign * float(cos)
        # aux["loss"] is the scalar semigreedy argmins over (design.py:465) and that
        # design.py:114 copies into aux["log"]["loss"] for save_best / mcmc.
        self.aux["loss"] = float(self.aux["loss"]) + weight * penalty
        # for logging/plots only; NOT read by the jitted graph.
        try:
            self.aux["losses"][key] = penalty
        except Exception:
            pass
        state["n"] += 1
        if verbose_first and state["n"] == 1:
            print(f"[balm] post-callback live: binder='{binder}' cos={cos:+.4f} "
                  f"penalty={penalty:+.4f} weight={weight} "
                  f"(off-graph; no pure_callback)")

    post_callback._balm_key = key
    post_callback.state = state
    af._callbacks["design"]["post"].append(post_callback)
    return post_callback


def remove_balm_design_callback(af, cb):
    """Detach. Cheap and safe -- design callbacks are not part of the jitted graph,
    so attaching/detaching never triggers a recompile."""
    try:
        af._callbacks["design"]["post"].remove(cb)
    except ValueError:
        pass


def calls(cb):
    """How many times the hook fired (once per candidate evaluation)."""
    return cb.state["n"]
