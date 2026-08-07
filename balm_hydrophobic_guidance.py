"""
balm_hydrophobic_guidance.py
============================
Hydrophobicity "walls" for the BALM-guided BindCraft Stage-4 search, in the same
idiom as the pLDDT/i_ptm structure barrier (loss_fn(inputs, outputs) -> {name: value};
opt["weights"][name] = weight). Attach right before Stage 4, next to
add_structure_barrier_loss.

Two options:

  A) add_hydrophobic_fraction_barrier_loss  -- simple cap on the TOTAL fraction of
     hydrophobic residues in the binder sequence (default max 0.50). Sequence-only,
     no structure needed.

  B) add_interface_hydrophobic_barrier_loss -- cap on the estimated hydrophobic
     surface that SURVIVES ProteinMPNN. With mpnn_fix_interface=True, MPNN keeps the
     trajectory's interface residues and polarizes the rest, so only the INTERFACE
     hydrophobics are "unavoidable". This term estimates
        (exposed hydrophobic SASA of interface residues) / (total exposed SASA)
     and walls it at a threshold (default 0.30-0.35). Non-interface residues are
     assumed polarized by MPNN and contribute 0 to the hydrophobic numerator.

     Interface membership is read from the AF2 structure via **Cbeta-Cbeta contacts**
     (binder CB within contact_dist of any target CB; falls back to CA for Gly). CB is
     used, not CA, because the backbone CA of a groove-buried residue sits ~8-12 A from
     the target CA even when its side chain is fully buried -- a CA cutoff misses real
     interface residues and the term collapses to ~0. Per-residue area uses reference
     max-SASA (Tien et al. 2013); interface residues are treated as partly buried
     (buried_frac, default 0.5). Intended for a single-helix binder.

Both are differentiable and cheap. Weight 100 (linear hinge) = a firm wall; the term
is 0 while the design is under the threshold.
"""
from __future__ import annotations

# AlphaFold/ColabDesign restype order -- must match inputs["seq"] one-hot columns.
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
# Residues counted as hydrophobic (edit to taste; C is usually omit_AAs anyway).
HYDROPHOBIC = set("AILMFWVYC")
# Tien et al. 2013 theoretical per-residue max SASA (A^2).
_MAXSASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"Q":225,"E":223,"G":104,"H":224,
            "I":197,"L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,
            "Y":263,"V":174}


def _arrays(jnp):
    hyd = jnp.asarray([1.0 if a in HYDROPHOBIC else 0.0 for a in RESTYPES])
    sasa = jnp.asarray([float(_MAXSASA[a]) for a in RESTYPES])
    return hyd, sasa


def _binder_onehot(inputs, blen, jnp):
    """(blen, 20) one-hot/soft sequence for the binder (last blen residues)."""
    seq = inputs["seq"]
    if isinstance(seq, dict):
        seq = seq.get("hard", seq.get("pseudo", next(iter(seq.values()))))
    seq = jnp.asarray(seq)
    if seq.ndim == 3:      # (num_seq, L, 20) -> first sequence
        seq = seq[0]
    return seq[-blen:]


# --------------------------------------------------------------------------- #
# A. Total hydrophobic-fraction wall
# --------------------------------------------------------------------------- #
def add_hydrophobic_fraction_barrier_loss(self, weight=100.0, max_frac=0.50,
                                          binder_len=None, key="hphob_frac"):
    """Penalize a binder whose fraction of hydrophobic residues exceeds max_frac."""
    blen = int(binder_len) if binder_len is not None else int(getattr(self, "_binder_len"))

    def loss_fn(inputs, outputs):
        import jax.numpy as jnp
        hyd, _ = _arrays(jnp)
        b = _binder_onehot(inputs, blen, jnp)          # (blen, 20)
        frac = jnp.sum(b @ hyd) / blen                 # fraction hydrophobic
        return {key: jnp.maximum(0.0, frac - max_frac)}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"][key] = weight
    return loss_fn


# --------------------------------------------------------------------------- #
# B. Interface hydrophobic exposed-SASA-fraction wall (post-MPNN estimate)
# --------------------------------------------------------------------------- #
def add_interface_hydrophobic_barrier_loss(self, weight=100.0, threshold=0.30,
                                           contact_dist=8.0, buried_frac=0.5,
                                           binder_len=None, target_len=None,
                                           key="hphob_iface"):
    """Wall on estimated (exposed hydrophobic SASA of interface residues)/(total exposed
    SASA). Interface = binder CB within contact_dist (A) of any target CB. buried_frac:
    fraction of an interface residue's surface taken as buried against the target."""
    blen = int(binder_len) if binder_len is not None else int(getattr(self, "_binder_len"))
    tlen = int(target_len) if target_len is not None else int(getattr(self, "_target_len"))

    def loss_fn(inputs, outputs):
        import jax
        import jax.numpy as jnp
        from colabdesign.af.alphafold.common import residue_constants
        CA = residue_constants.atom_order["CA"]
        CB = residue_constants.atom_order["CB"]
        hyd, sasa = _arrays(jnp)
        b = _binder_onehot(inputs, blen, jnp)                 # (blen, 20)
        hyd_i = b @ hyd                                       # (blen,) 1 if hydrophobic
        sasa_i = b @ sasa                                     # (blen,) max SASA
        # Cbeta contact points (fall back to CA where CB is absent, e.g. Gly)
        atoms = outputs["structure_module"]["final_atom_positions"]   # (Ltot, 37, 3)
        ca = atoms[:, CA]
        cb = atoms[:, CB]
        no_cb = (jnp.abs(cb).sum(-1) < 1e-6)                  # Gly / missing CB
        rep = jnp.where(no_cb[:, None], ca, cb)              # (Ltot, 3) side-chain proxy
        tgt = rep[:tlen]
        bnd = rep[-blen:]
        mind = jnp.min(jnp.linalg.norm(bnd[:, None, :] - tgt[None, :, :], axis=-1), axis=1)  # (blen,)
        iface = jax.nn.sigmoid(contact_dist - mind)          # soft 0..1 (per binder residue)
        exposed = sasa_i * (1.0 - buried_frac * iface)       # interface residues partly buried
        num = jnp.sum(iface * hyd_i * exposed)               # exposed hydrophobic area of interface res
        den = jnp.sum(exposed) + 1e-8                        # total exposed surface
        frac = num / den
        return {key: jnp.maximum(0.0, frac - threshold)}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"][key] = weight
    return loss_fn
