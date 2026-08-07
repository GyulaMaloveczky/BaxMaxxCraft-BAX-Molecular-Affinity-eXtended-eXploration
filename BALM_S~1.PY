"""
balm_structure_guidance.py
==========================
Keeps a BALM-guided BindCraft Stage-4 search ON the foldable, natural-protein
manifold while BALM pushes the sequence OFF AF2's confidence optimum.

Two extra ColabDesign loss terms, attached with the same idiom as add_balm_loss
(loss_fn(inputs, outputs) -> {name: value}; opt["weights"][name] = weight):

  1) add_structure_barrier_loss  -- a HARD FLOOR on binder pLDDT and interface
     pTM. Differentiable (no stop_gradient). Silent above the floors; a steep
     linear wall below them. This is the key fix: instead of letting BALM trade
     structure away in a soft weighted sum (which collapsed folds to plddt~0.35),
     the barrier confines the search to sequences AF2 still folds confidently,
     and BALM only moves you *laterally* within that set.

  2) add_esm_naturalness_loss    -- OPTIONAL floor on ESM sequence log-likelihood
     so BALM can't win by proposing bizarre-but-foldable sequences that drift
     out of the distribution where its cosine still predicts binding. External
     PyTorch scorer -> pure_callback + stop_gradient (value-only, which is what
     the discrete Stage-4 search consumes), same as BALM.

attach_guided_balm() wires all three (BALM + barrier + naturalness) in one call.

ATTACH POINT: exactly where you attach add_balm_loss today -- right before the
Stage-4 design_pssm_semigreedy call, after prep_inputs() has set self._binder_len.
"""
from __future__ import annotations

# AlphaFold/ColabDesign canonical residue order: "".join(residue_constants.restypes)
RESTYPES = "ARNDCQEGHILKMFPSTWYV"


def _indices_to_sequence(idx, alphabet=RESTYPES):
    return "".join(alphabet[int(i)] if 0 <= int(i) < len(alphabet) else "X" for i in idx)


# --------------------------------------------------------------------------- #
# 1. Structural barrier (differentiable): floor on binder pLDDT and interface pTM
# --------------------------------------------------------------------------- #
def add_structure_barrier_loss(self, weight=100.0, plddt_floor=0.8, iptm_floor=0.8,
                               binder_len=None, key="struct_barrier"):
    """Append a one-sided barrier that penalizes dropping below the confidence floors.

    weight       : opt weight. 100 with a linear hinge makes a hard wall -- e.g. a
                   0.05 dip below a floor already costs 100*0.05 = 5 loss, which
                   dominates any realistic BALM per-step gain. Raise for a harder wall.
    plddt_floor  : minimum mean binder pLDDT (0-1).   Default 0.8.
    iptm_floor   : minimum interface pTM (0-1).        Default 0.8.
    binder_len   : defaults to self._binder_len (binder = last binder_len residues).

    Contribution is 0 while both metrics are >= their floors, so above the floors
    the search is driven purely by BALM + the native BindCraft terms.
    """
    blen = int(binder_len) if binder_len is not None else int(getattr(self, "_binder_len"))

    def loss_fn(inputs, outputs):
        import jax.numpy as jnp
        try:
            from colabdesign.af.loss import get_plddt, get_ptm
        except Exception as e:  # pragma: no cover - version guard
            raise ImportError(
                "balm_structure_guidance: could not import get_plddt/get_ptm from "
                "colabdesign.af.loss; adjust the import to match your ColabDesign build."
            ) from e
        plddt = get_plddt(outputs)                          # (L,) in [0,1]
        binder_plddt = jnp.mean(jnp.asarray(plddt)[-blen:])  # binder = last blen residues
        iptm = get_ptm(inputs, outputs, interface=True)      # scalar in [0,1]
        viol = (jnp.maximum(0.0, plddt_floor - binder_plddt)
                + jnp.maximum(0.0, iptm_floor - iptm))
        return {key: viol}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"][key] = weight
    return loss_fn


# --------------------------------------------------------------------------- #
# 2. ESM naturalness floor (optional external scorer)
# --------------------------------------------------------------------------- #
class EsmNaturalness:
    """Mean per-residue ESM log-likelihood of a sequence (higher = more natural).

    Uses a single unmasked forward pass through an ESM MLM head -- cheap (one pass)
    and enough to flag adversarial, unnatural sequences. Lazy-loads torch.
    """

    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", device="cuda"):
        self.model_name = model_name
        self.device = device
        self._tok = None
        self._model = None
        self._dev = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            dev = self.device
            if dev.startswith("cuda") and not torch.cuda.is_available():
                print("[naturalness] cuda unavailable; using CPU.")
                dev = "cpu"
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(dev).eval()
            self._dev = dev
        return self._tok, self._model

    def mean_logprob(self, seq: str) -> float:
        import torch
        tok, model = self._ensure()
        with torch.no_grad():
            enc = tok([seq], return_tensors="pt")
            enc = {k: v.to(self._dev) for k, v in enc.items()}
            logits = model(**enc).logits[0]                      # (L+special, vocab)
            logp = torch.log_softmax(logits, dim=-1)
            ids = enc["input_ids"][0]
            tok_logp = logp[torch.arange(ids.shape[0], device=self._dev), ids]
            special = torch.tensor(
                tok.get_special_tokens_mask(ids.tolist(), already_has_special_tokens=True),
                device=self._dev, dtype=torch.bool)
            return float(tok_logp[~special].mean().item())       # <= 0, closer to 0 = natural


def add_esm_naturalness_loss(self, weight=1.0, model_name="facebook/esm2_t33_650M_UR50D",
                             device="cuda", floor=None, margin=1.0, binder_len=None,
                             alphabet=RESTYPES, key="naturalness", scorer=None):
    """Append a one-sided naturalness floor. Penalizes only sequences that become
    much LESS natural than where the trajectory started.

    floor  : absolute log-likelihood floor. If None (default), it is set ADAPTIVELY
             from the first sequence seen: floor = firstseq_logprob - margin. i.e.
             'don't let the binder get more than `margin` nats/residue less likely
             than the design you started Stage 4 with.'
    margin : slack for the adaptive floor (nats/residue). Default 1.0.
    weight : opt weight. 1.0 is a gentle regularizer; the hinge is 0 for normal seqs.
    """
    import numpy as np
    blen = int(binder_len) if binder_len is not None else int(getattr(self, "_binder_len"))
    scorer = scorer or EsmNaturalness(model_name, device)
    state = {"floor": floor}

    def host_penalty(seq_idx_np):
        idx = np.asarray(seq_idx_np).reshape(-1)[-blen:]
        seq = _indices_to_sequence(idx, alphabet)
        nat = scorer.mean_logprob(seq)                 # per-residue mean log P (<= 0)
        if state["floor"] is None:                     # calibrate off the first sequence
            state["floor"] = nat - float(margin)
        viol = max(0.0, state["floor"] - nat)          # bites only when much less natural
        return np.asarray(viol, dtype=np.float32)

    def loss_fn(inputs, outputs):
        import jax
        import jax.numpy as jnp
        seq = inputs["seq"]
        if isinstance(seq, dict):
            seq = seq.get("hard", seq.get("pseudo", next(iter(seq.values()))))
        seq = jnp.asarray(seq)
        if seq.ndim == 3:
            seq = seq[0]
        seq_idx = jnp.argmax(seq, axis=-1)
        pen = jax.pure_callback(
            host_penalty, jax.ShapeDtypeStruct((), jnp.float32), seq_idx,
            vmap_method="sequential",   # drop this kwarg on jax < 0.4.24
        )
        return {key: jax.lax.stop_gradient(pen)}

    loss_fn.host_penalty = host_penalty
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"][key] = weight
    return loss_fn


# --------------------------------------------------------------------------- #
# 3. One-call wiring: BALM + barrier + (optional) naturalness
# --------------------------------------------------------------------------- #
def attach_guided_balm(self, aux_loss, balm_weight, *, binder_len=None,
                       barrier_weight=100.0, plddt_floor=0.8, iptm_floor=0.8,
                       naturalness=True, naturalness_weight=1.0,
                       naturalness_model="facebook/esm2_t33_650M_UR50D",
                       naturalness_margin=1.0, device="cuda"):
    """Attach BALM + structure barrier (+ optional ESM naturalness) before Stage 4.

    Returns {name: loss_fn}. Replaces your single add_balm_loss(...) call.
    """
    from balm_colabdesign_bridge import add_balm_loss
    blen = int(binder_len) if binder_len is not None else int(getattr(self, "_binder_len"))
    fns = {}
    fns["balm"] = add_balm_loss(self, aux_loss, balm_weight, binder_len=blen)
    fns["struct_barrier"] = add_structure_barrier_loss(
        self, weight=barrier_weight, plddt_floor=plddt_floor,
        iptm_floor=iptm_floor, binder_len=blen)
    if naturalness:
        fns["naturalness"] = add_esm_naturalness_loss(
            self, weight=naturalness_weight, model_name=naturalness_model,
            device=device, margin=naturalness_margin, binder_len=blen)
    return fns
