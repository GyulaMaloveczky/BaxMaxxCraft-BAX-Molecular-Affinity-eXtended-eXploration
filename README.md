# BaxMaxxCraft

**BAX Molecular Affinity eXtended eXploration**. A fork of
[BindCraft](https://github.com/martinpacesa/BindCraft) that adds a learned
affinity model to the discrete design search.

Stock BindCraft selects binders on AlphaFold2 confidence. On the BAX canonical
groove those metrics track physics-based binding energy weakly: interface pTM
explains 6% of the variance in MM/PBSA and, at fixed binder length, the
composite BindCraft objective does not correlate with it. This fork adds an
external affinity signal. A BALM-PPI model, fine-tuned on MM/PBSA labels from
our own MD simulations, scores candidate sequences inside stage 4 and biases the
search towards stronger predicted binders.

Part of an MSc thesis project at the University of Amsterdam and Vrije
Universiteit Amsterdam (computational chemistry group, supervisor Dr. ir.
I. M. Ilie). The analysis code is in a
[separate repository](https://github.com/GyulaMaloveczky/BAXMAXX_analysis).

![Design, simulate and refit loop. BindCraft generates and filters binders; each
accepted design is simulated and scored by MM/PBSA; the labels refit a
sequence-based affinity model; the refitted model is added to the discrete
search stage of the next campaign.](figures/fig_workflow.png)

*The loop this fork implements. Step 2 costs hours of compute per design; step 4
calls the affinity model hundreds of times per trajectory at negligible cost.
This fork is step 4.*

---

## What is different from upstream BindCraft

| Change | Where |
|---|---|
| Affinity score added to the stage-4 semi-greedy search | `balm_design_callback.py` |
| Hard floors on pLDDT and interface pTM | `balm_structure_guidance.py` |
| ESM2 sequence-likelihood floor | `balm_structure_guidance.py` |
| Hydrophobic-fraction cap | `balm_hydrophobic_guidance.py` |
| Four-stage protocol (PSSM semi-greedy stage added) | `functions/colabdesign_utils.py` |
| Per-candidate recording for later analysis | `balm_candidate_recorder.py` |
| Adversarial/augmented sampling | `functions/augment_sampling.py`, `bindcraft_augment.py` |

`bindcraft_original.py` is kept unmodified for diffing.

### Why a design post-callback, not a loss term

The direct approach is to append the affinity term to
`_callbacks["model"]["loss"]`. That function is traced into the jitted
AlphaFold2 graph, so reaching an external PyTorch model from inside it requires
`jax.pure_callback`, which forces execution out of the compiled graph on every
evaluation. Measured on a single trajectory, stage 4 rose from about 69 s to
292 s for 1.3 s of model compute, and compilation time rose roughly sixfold.

The semi-greedy search accepts or rejects candidates on a plain Python scalar,
so the score is instead attached as a ColabDesign **design** post-callback. It
fires once per candidate rather than once per model per recycle, reducing
evaluations per trajectory from about 404 to 100.

### Why the hard floors exist

![MM/PBSA against the design-time metrics. Interface area correlates at r =
-0.85; at fixed binder length interface pTM reaches r = -0.03 and the full
BindCraft objective r = 0.09.](figures/fig_correlates.png)

*Why an external affinity signal is needed at all. At fixed binder length the
composite BindCraft objective does not correlate with simulated binding energy.*

Weighting the affinity score against the native BindCraft loss was not
sufficient. The search improved predicted affinity while driving binder pLDDT
down to roughly 0.35, into sequences AlphaFold2 no longer predicted would fold.
This is the expected behaviour of a semi-greedy search given an imperfect proxy
and enough steps. No weighting avoided it: any weight large enough to steer the
search was also large enough to allow structure to be traded away.

The floors constrain the search space rather than rebalancing the objective, so
the affinity term can only move the sequence laterally among candidates that
remain foldable. **The floors are attached independently of the affinity term**,
so an unguided control shares the same loss surface, gating and acceptance rules
and differs only in the absence of the score. Setting `weight: 0` therefore
gives a valid control.

---

## Installation

Identical to upstream. Requires a CUDA-capable GPU.

```bash
git clone https://github.com/GyulaMaloveczky/BaxMaxxCraft-BAX-Molecular-Affinity-eXtended-eXploration [install_folder]
cd [install_folder]
bash install_bindcraft.sh --cuda '12.4' --pkg_manager 'conda'
```

The installer pulls PyRosetta, which needs a licence for commercial use, and
about 5.3 GB of AlphaFold2 weights.

Additionally required for the affinity model: `torch`, `transformers`, `peft`,
and a BALM-PPI checkpoint. Point `balm_config.json` at the checkpoint.

## Running

```bash
python bindcraft.py \
  --settings       settings_target/<your_target>.json \
  --filters        settings_filters/peptide_relaxed_filters.json \
  --advanced       settings_advanced/peptide_4stage_multimer.json \
  --balm_config    balm_config.json
```

On a cluster, see `bindcraft.slurm`.

Target, filter and advanced settings work exactly as upstream; see the
[BindCraft wiki](https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft).

## Configuring the affinity guidance

`balm_config.json` holds everything specific to this fork.

```jsonc
{
  "balm_aux": {
    "enabled": true,
    "weight": 20.0,          // scales the score's contribution
    "direction": "minimize", // checkpoint-dependent; see below
    "metric": "cosine",      // bounded [-1, 1]
    "active_stages": [4],    // discrete stages only
    "target_sequence": "..." // fixed receptor
  },
  "structure_barrier": {
    "enabled": true,
    "weight": 100.0,
    "plddt_floor": 0.75,
    "iptm_floor": 0.75       // silent above the floors
  },
  "esm_naturalness": {
    "enabled": true,
    "weight": 1.0,
    "margin": 1.0
  }
}
```

**`direction` is checkpoint-dependent.** Some checkpoints score a stronger
binder as a more negative cosine. Confirm which convention a checkpoint uses
before running a campaign: an inverted setting steers the search away from
strong binders while producing output that appears normal.

`BALM_AUX_README.md` documents the loss adapter in more detail.

## Repository layout

```
balm_*.py                 affinity model, loss adapter, guidance barriers
bindcraft.py              main entry point (modified)
bindcraft_original.py     upstream, unmodified, for diffing
bindcraft_augment.py      adversarial / augmented sampling run
functions/                pipeline internals; colabdesign_utils.py holds the
                           four-stage protocol and barrier attachment
settings_advanced/        design protocols
settings_filters/         filter panels
settings_target/          target definitions
notebooks/                Colab notebook (upstream)
test_balm_integration.py  integration test for the affinity coupling
verify_balm_on_testset.py checkpoint sanity check against a held-out set
```

## Status

See `TODO.md`. The main outstanding item is a train/deploy mismatch: the
affinity model is trained on post-MPNN sequences but applied to pre-MPNN ones
during stage 4. The two correlate at r = 0.76.

## Licence and attribution

Upstream BindCraft is by Martin Pacesa et al.
([paper](https://www.nature.com/articles/s41586-025-09429-6),
[preprint](https://www.biorxiv.org/content/10.1101/2024.09.30.615802)); see
`LICENSE`. Please cite BindCraft if you use this fork. BALM-PPI, ESM2,
ProteinMPNN/SolubleMPNN, AlphaFold2 and PyRosetta retain their own licences and
citation requirements.
