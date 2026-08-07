"""
balm_contact_guidance.py
========================
Interface CONTACT barrier: a hard wall against the "floating peptide" failure mode
where AF2 reports a confident interface (i_ptm ~0.87, i_con ~5) while no binder atom
is actually within 5 A of the target -- so `hotspot_residues()` returns {} and the
trajectory is discarded before MPNN.

Root cause it addresses: `i_con` counts a contact at CB-CB <= 10 A (opt["i_con"]
["cutoff"]), whereas hotspot_residues counts ANY atom pair <= 5 A. A peptide hovering
7-8 A off the surface satisfies i_con completely and fails the interface filter. The
existing walls constrain pLDDT / i_ptm / hydrophobicity but nothing enforces contact,
and BALM scores sequence only -- so nothing in the search objected.

This measures the SAME quantity the filter does: true atom-atom distance at 5 A.

PENALTY SCHEDULE (steep wall, as specified)
    0 contacts -> +50
    1 contact  -> +10
    2 contacts -> + 5
    >=3        ->   0
Interpolated with jnp.interp, so with a soft count the ramp is smooth (and therefore
usable in the gradient stages); with an exact integer count it reproduces the step
values above exactly.

COST
    binder_atoms x target_atoms = (blen*37) x (tlen*37).
    For blen=15, tlen~200 that is ~4.1M distances (~16 MB) -- negligible against an
    AF2 forward pass. Pure JAX: no pure_callback, no device stall.
"""
from __future__ import annotations

DEFAULT_SCHEDULE = (50.0, 10.0, 5.0)     # penalty at 0, 1, 2 contacts; 0 at >= min_contacts


def add_interface_contact_barrier_loss(self, weight=1.0, atom_cutoff=5.0, min_contacts=3,
                                       schedule=DEFAULT_SCHEDULE, soft=True, temp=0.25,
                                       key="i_contact", binder_len=None, target_len=None):
    """Wall the search off from non-contacting solutions.

    self         : af_model from mk_afdesign_model(protocol="binder").
    weight       : registered as opt["weights"][key]. Keep at 1.0 so the schedule
                   values ARE the score increments (unlike the other barriers, which
                   use weight=100 against a 0-1 loss).
    atom_cutoff  : Angstroms, atom-atom. 5.0 matches hotspot_residues' default.
    min_contacts : binder residues in contact required to zero the penalty.
    schedule     : penalty at 0,1,2,... contacts. len(schedule) should be min_contacts.
    soft         : count contacts with a sigmoid instead of a step, so the term has a
                   gradient. temp is the sigmoid width in Angstroms; 0.25 is nearly a
                   step (a residue at 4.9 A counts ~1.0, at 5.1 A ~0.0).
    """
    import jax
    import jax.numpy as jnp

    blen = int(binder_len) if binder_len is not None else int(getattr(self, "_binder_len"))
    tlen = int(target_len) if target_len is not None else int(getattr(self, "_target_len"))
    xp = jnp.arange(len(schedule) + 1, dtype=jnp.float32)          # [0,1,2,3]
    fp = jnp.array(tuple(schedule) + (0.0,), dtype=jnp.float32)    # [50,10,5,0]

    def loss_fn(inputs, outputs):
        sm = outputs["structure_module"]
        pos = sm["final_atom_positions"]                      # (L, 37, 3)
        # Unused atom slots (glycine CB, absent side chains) sit at the origin.
        # WITHOUT masking they manufacture phantom contacts at ~0 A.
        if "final_atom_mask" in sm:
            mask = sm["final_atom_mask"]                      # (L, 37)
        else:
            mask = (jnp.abs(pos).sum(-1) > 0).astype(pos.dtype)

        t_pos, t_mask = pos[:tlen], mask[:tlen]               # target = leading residues
        b_pos, b_mask = pos[-blen:], mask[-blen:]             # binder = trailing residues

        t_flat = t_pos.reshape(-1, 3)                         # (tlen*37, 3)
        t_mflat = t_mask.reshape(-1)
        b_flat = b_pos.reshape(blen, -1, 3)                   # (blen, 37, 3)
        b_mflat = b_mask.reshape(blen, -1)                    # (blen, 37)

        # (blen, 37, tlen*37) pairwise distances
        d = jnp.linalg.norm(b_flat[:, :, None, :] - t_flat[None, None, :, :], axis=-1)
        valid = (b_mflat[:, :, None] > 0) & (t_mflat[None, None, :] > 0)
        d = jnp.where(valid, d, jnp.inf)

        # closest approach of each BINDER RESIDUE to any target atom
        min_d = d.reshape(blen, -1).min(-1)                   # (blen,)

        if soft:
            # sigmoid((cutoff - d)/temp): ~1 inside the cutoff, ~0 outside, differentiable
            contacts = jax.nn.sigmoid((atom_cutoff - min_d) / temp).sum()
        else:
            contacts = (min_d < atom_cutoff).sum().astype(jnp.float32)

        # steep wall: 0->50, 1->10, 2->5, >=min_contacts->0
        penalty = jnp.interp(contacts, xp, fp, right=0.0)
        return {key: penalty}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"][key] = weight
    return loss_fn
