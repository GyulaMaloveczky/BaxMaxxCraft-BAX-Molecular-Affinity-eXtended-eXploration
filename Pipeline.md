# Running the affinity-guided design pipeline

How to reproduce the loop end to end, or point it at a new target.

The loop has four stages. Stage 2 is the expensive one, at hours of compute per
design. Stages 1 and 4 are the same program with different settings: stage 4 is
stage 1 with an affinity model attached.

```
  1. BindCraft generation  ──►  2. MD + MM/PBSA  ──►  3. Fine-tune BALM-PPI
                                   ▲                            │
                                   └─ 4. Guided generation ◄────┘
```

> **Placeholders.** `<...>` marks a path or name you must fill in for your own
> setup. Everything else is as it ran for the BAX campaigns.

---

## Repositories and environments

| What | Where |
|---|---|
| Design pipeline (BindCraft fork) | [BaxMaxxCraft](https://github.com/GyulaMaloveczky/BaxMaxxCraft-BAX-Molecular-Affinity-eXtended-eXploration) |
| Analysis notebooks | [BAXMAXX_analysis](https://github.com/GyulaMaloveczky/BAXMAXX_analysis) |
| BALM-PPI training code | `<REPO_OR_PATH>` |

| Environment | Used for | Created by |
|---|---|---|
| `<BINDCRAFT_ENV>` | stages 1 and 4 | `install_bindcraft.sh` |
| `<MD_ENV>` | stage 2, GROMACS + gmx_MMPBSA + AmberTools | `<HOW>` |
| `<BALM_ENV>` | stage 3, PyTorch + transformers + peft | `<HOW>` |
| `<ANALYSIS_ENV>` | notebooks | `<HOW>` |

Install the design pipeline:

```bash
git clone https://github.com/GyulaMaloveczky/BaxMaxxCraft-BAX-Molecular-Affinity-eXtended-eXploration bindcraft
cd bindcraft
bash install_bindcraft.sh --cuda '12.4' --pkg_manager 'conda'
```

This pulls PyRosetta, which needs a licence for commercial use, and about 5.3 GB
of AlphaFold2 weights. A CUDA-capable GPU is required.

Set the data root once so the notebooks resolve paths (see `paths.py` in the
analysis repo):

```bash
export BAXMAXX_DATA=<DATA_ROOT>        # one subdirectory per campaign
export BAXMAXX_MODELS=<MODEL_ROOT>     # training runs and checkpoints
```

---

## 1. BindCraft generation

Produces designed sequences and their predicted complexes. This is the unguided
pass that creates the initial training set.

### Inputs

| File | Purpose |
|---|---|
| `settings_target/<target>.json` | target PDB, chain, hotspot residues, binder length |
| `settings_filters/peptide_relaxed_filters.json` | filter panel |
| `settings_advanced/<protocol>.json` | stage schedule, loss weights |

The target JSON is the file you edit for a new target. It names the structure,
the chain, the hotspot residue range and the binder length. For BAX this was
residues 74 to 99 of the open full-length model, binder length 15.

> **Define the target region generously.** Residues 74 to 99 cover the central
> part of the BAX canonical groove, where alpha-9 and BH3 helices make their
> closest contacts, but the groove as usually defined extends further along the
> same face. The window was inherited from earlier work in the group and its
> original rationale is not recorded.
>
> This appears to have cost us. The strongest designs by MM/PBSA tend to extend
> past the specified window and pick up contacts on the far side of the groove,
> which the search was never rewarded for finding. In other words the optimum
> sat partly outside the region we told BindCraft to optimise inside.
>
> Verify this on your own data before relying on it: take the top decile by
> MM/PBSA, compute per-target-residue contact frequency, and compare against the
> specified window. If strong binders systematically contact residues outside
> it, widen the region.
>
> The asymmetry argues for erring wide. Too narrow a region forecloses poses the
> search can never recover. Too wide mostly costs yield, because more
> trajectories fail the interface checks. Yield is cheaper than a foreclosed
> optimum.

### Run

```bash
conda activate <BINDCRAFT_ENV>
python bindcraft.py \
  --settings    settings_target/<target>.json \
  --filters     settings_filters/peptide_relaxed_filters.json \
  --advanced    settings_advanced/<protocol>.json
```

On a cluster use `bindcraft.slurm` as a template.

### Settings worth changing

| Setting | Default here | When to change |
|---|---|---|
| Binder length | 15 | Fix it. Length dominates MM/PBSA, so a variable-length campaign creates a confound you then have to correct for in training. Vary it only if you deliberately want a length-sweep campaign. |
| Hotspot residues | 74--99 | Always, for a new target. Pick the surface the interface must form on, and see the note above on defining it generously. |
| Stage-4 candidates per step | 5 | More candidates means a better search but linearly more AlphaFold2 passes. BindCraft's default is 1. |
| Stage-4 steps | 20 or 50 | 50 for final campaigns, 20 when exploring. |
| Helicity weight | 0.95 | Lower if the target site does not demand a helix. |
| Filter panel | `peptide_relaxed_filters` | Tightening the filters reduces yield sharply at this binder length. |

### Outputs

```
<CAMPAIGN_DIR>/
  Accepted/                  designed complexes, PDB
  mpnn_design_stats.csv      per-design metrics after MPNN
  trajectory_stats.csv       per-trajectory record, including stage-4 path
```

Both CSVs feed the analysis notebooks. `mpnn_design_stats.csv` carries the
interface metrics (dSASA, interface pTM, pLDDT) used in stage 3 diagnostics.

> **Keep the whole campaign, not just the best designs.** The affinity model is
> trained on the spread. Discarding weak designs removes the examples that teach
> it what weak looks like.

---

## 2. Molecular dynamics and MM/PBSA

Turns each accepted design into a binding-energy label. This is the only part of
the loop that costs real compute.

### Preparation

Truncate flexible terminal regions distant from the interface and cap the
resulting chain ends with acetyl and N-methylamide groups, so they carry no
artificial formal charge. This shrinks the water box considerably. AlphaFold2 is
insensitive to chemical caps, so this does not affect the design stage, only the
simulation.

Scripts: `<PREP_SCRIPT>`

### Simulation protocol

GROMACS with CHARMM36 and CHARMM-modified TIP3P water, 2 fs timestep.

| Stage | Ensemble | Length | Restraint on C-alpha |
|---|---|---|---|
| Minimisation | — | until F_max < 1000 kJ/mol/nm or 10,000 steps | — |
| Equilibration 1 | NPT | 1 ns | 1000 kJ/mol/nm² |
| Equilibration 2 | NPT | 1 ns | 100 kJ/mol/nm² |
| Settling | NVT | 1 ns | none |
| Production | NVT | 10 ns × 5 replicates | none |

Temperature 310 K by velocity rescaling (tau_t = 2.0 ps). Pressure 1.0 bar by
Berendsen barostat (tau_p = 2.0 ps). PME electrostatics, 1.2 nm real-space
cut-off, 0.16 nm grid. Van der Waals 1.2 nm with potential-shift under Verlet.
Bonds to hydrogen constrained with LINCS. Solvated in a periodic box,
neutralised, 150 mM NaCl.

The five replicates differ only in initial velocities.

MDP files: `<MDP_DIR>`
Submission: `<MD_SUBMIT_SCRIPT>`

### MM/PBSA

`gmx_MMPBSA` under a single-trajectory protocol: complex, free receptor and free
peptide all extracted from the same simulation.

| Parameter | Value |
|---|---|
| Interior dielectric | 1.5 |
| Exterior dielectric | 80.0 |
| Ionic strength | 0.15 M |
| Probe radius | 1.4 Å |
| Surface tension | 0.0072 kcal/mol/Å² |
| Radii | CHARMM-derived |
| Frames | 80 per replicate, 400 total per design |
| Entropy | not included |

Input file: `<MMPBSA_IN>`

**The interior dielectric is the one parameter you should recalibrate for a new
target.** Values of 1, 1.5, 2, 3, 4 and 6 were tested, and the one retained was
whichever correlated most strongly, by Pearson correlation, with umbrella-sampling
binding free energies for four benchmark complexes. Run
`Umbrella-MMPBSA.ipynb` to redo this. Without reference free energies for your
target, 1.5 is a defensible starting point, but say so.

**Entropy is deliberately omitted.** Including a configurational-entropy term
worsened agreement with the umbrella-sampling reference. The consequence is that
reported energies are systematically more favourable than true binding free
energies. They are a ranking signal, not a free energy, and should never be
quoted on a physical scale.

**400 frames, not 2000.** Replicate means use the same total number of frames as
a single-trajectory label. The replicates buy reliability, not sampling depth.

### Extracting labels

| Notebook | What it does |
|---|---|
| `MMPBSA_MD_analysis.ipynb` | Main processing of gmx_MMPBSA output into per-design energies |
| `MMPBSA_new_binders.ipynb` | Same, for a later campaign |
| `All_mmpbsas.ipynb` | Pools every campaign into one table |
| `RMSD_adder.ipynb` | Adds binder Cα RMSD as a pose-stability diagnostic |
| `Umbrella-MMPBSA.ipynb` | Dielectric calibration against umbrella-sampling references |

Output table: `<LABEL_TABLE>`, one row per design with the MM/PBSA mean, the
per-replicate values and the replicate count.

### Replicates and reproducibility

Run `Fidelity_check.ipynb`. It produces `figures/fig_replicates.png` and the
numbers below.

Correlate all pairs of replicate slots across designs. Five replicates gives ten
slot pairs; correlate each slot's vector of MM/PBSA values against each other
slot's, across designs, and average the ten correlations.

```
r₁   = 0.815     single-run reliability
r_kk = 0.957     five-replicate mean, by Spearman–Brown
                 r_kk = k·r₁ / (1 + (k−1)·r₁)

ceiling against a single replicate      √0.815 = 0.903
ceiling against a five-replicate mean   √0.957 = 0.978
```

**Read every later correlation against 0.978.** A model at Pearson 0.47 is at 48%
of what the label permits, not 47% of perfect.

Averaging replicates assumes they sample one basin with independent noise, rather
than drifting into distinct poses. Check this, do not assume it:

- Per-replicate agreement with the mean of the other four should sit around
  0.85–0.91.
- Overlay the dihedral-PCA centroid frame of each replicate for a few designs and
  confirm they superpose.
- **Inspect the binder RMSD traces.** Discrete steps indicate a transition between
  metastable states within a run. For those designs the label is an average over
  more than one pose, not a measurement of one. Split the trajectory at the step
  and score each segment separately to see how much it costs.

Re-run `Fidelity_check.ipynb` for a new target. The reliability is a property of
the system, not a constant.

---

## 3. Fine-tuning BALM-PPI

Fits a sequence-only model to the MM/PBSA labels, so the expensive signal can be
consulted cheaply inside the search.

Code: `<BALM_REPO>`

### Data preparation

| Notebook | What it does |
|---|---|
| `PPB-Affinity_peptides.ipynb` | Prepares PPB-Affinity for pretraining; enforces longer-chain-first ordering |
| `PPB-Affinity Augment.ipynb` | Augmented pretraining set |
| `Clear_BCL2_from_train.ipynb` | Removes BCL-2 family entries from pretraining, to test memorisation |
| `Prepare_training data.ipynb` | Builds the fine-tuning set from the MM/PBSA labels |

### Pretraining

On PPB-Affinity: 12,062 measured protein-protein affinities. Entries with more
than one chain per side removed. Each pair ordered so the longer chain always
enters the same encoder. Labels min-max scaled to [-1, 1].

The published cross-validation split is too permissive. Rebuild it: pool and
deduplicate sequences, compare by truncation-tolerant BLOSUM62 alignment with
free end gaps, cluster by average linkage on 1 − similarity at 0.6, then treat
clusters as graph nodes and assign whole connected components to five folds,
largest first into the lightest fold. No measured pair then straddles the
train/test boundary.

Config: `<PRETRAIN_CONFIG>`

| Setting | Value |
|---|---|
| Encoder | ESM2 650M |
| LoRA | query and value, blocks 29–32 |
| Projection dimension | 128 |
| Dropout | 0.3 |
| Batch size | 16 |
| Learning rate | 1e-4, 6% linear warmup |
| Max epochs | 30, patience 10 |

### Fine-tuning

Config: `<FINETUNE_CONFIG>`

| Setting | Value | Note |
|---|---|---|
| Trained encoder | binder side only | The target never changes, so freeze and cache its representation |
| LoRA | rank 16, alpha 32, QKV | Separate from and larger than the pretraining adapter |
| Projection dimension | 256 | |
| Objective | smooth Kendall τ surrogate | Rank, do not regress |
| Batch size | 32, **length-stratified** | See below |
| Learning rate | 1e-4, warmup then decay | |
| Weight averaging | EMA, decay 0.99 | |
| Unrankable-pair cut | 0.3 kcal/mol | See improvements |
| Replicate weighting | 3.89× for five-replicate labels | Ratio of inverse noise variances |
| Cross-validation | 3 folds, homology-clustered at 0.80 | Trajectory siblings must not split across folds |
| Inner holdout | 15%, for early stopping | Keeps the outer fold clean |
| Deployed model | ensemble of 3 seeds | Each trained on the full set for the median epoch count its folds selected |

**Rank, do not regress.** A squared-error objective on noisy labels pulls
predictions toward the training mean. The search never needs a calibrated number,
only a reliable answer to whether one mutation beats another.

**Length-stratified batching is not optional.** With random batches the model
reaches a raw Kendall τ of +0.720, but a baseline that knows only binder length
reaches −0.769 by itself: the model is counting residues. Stratifying by length
makes within-batch length variance zero, so length cannot contribute to a
pairwise objective computed inside a batch. The raw τ drops to +0.222 while the
length-controlled τ rises from +0.234 to +0.256. **The headline number getting
worse is the correct outcome.**

### Did the training work?

Run `inspect_training.ipynb`. Check, in order:

1. **Stratified Kendall τ, not raw τ.** Only pairs of equal length count. A high
   raw τ with a low stratified τ means the model learned length.
2. **Pearson against the five-replicate labels**, read as a fraction of the 0.978
   ceiling. Model B reached 0.47 ± 0.06 over three seeds on n = 266, which is 48%
   of the ceiling.
3. **Seed-to-seed standard deviation.** It was 0.022 here. Any configuration
   change smaller than this is not a result.
4. **Decile means across the score range.** They should rise monotonically. If the
   association comes only from one end, the model separates weak from strong but
   does not order within the range the search actually occupies.
5. **Pre-MPNN versus post-MPNN agreement.** The model trains on post-MPNN
   sequences and deploys on pre-MPNN ones. These correlated at r = 0.76, with the
   fit departing from identity by +0.315 to −0.249 across the score range. Check
   this before deploying; if it degrades, the deployed score is measuring
   something else.

`verify_balm_on_testset.py` in the BaxMaxxCraft repo is a checkpoint sanity check
against a held-out set.

### Learning curve: how much more data is worth it

Subsample the training set, stratified by binder length so no length group falls
below the batching threshold, three draws per size.

| Training designs | Δτ vs previous | Cumulative |
|---|---|---|
| 200 | — | — |
| 400 | +0.086 | +0.086 |
| 600 | +0.040 | +0.126 |
| 900 | +0.032 | +0.158 |
| 1300 | +0.021 | +0.179 |

The last increment equals the seed noise. Extrapolating a power law, doubling the
dataset gives +0.02 to +0.04, which this evaluation cannot resolve.

**The practical reading: more designs approach the ceiling slowly, replicates
raise it.** If you have compute to spend, spend it on replicate coverage of
existing designs rather than on new designs.

A second reading worth knowing: the curve scales with designs pooled across the
full range of binder lengths, so off-length designs do contribute usable signal.
Keep them and neutralise length through batching rather than discarding them.

### Where the model could be improved

None of these were tested; they are the obvious next moves.

**The near-tie margin is absolute where it should be relative.** Pairs whose
labels differ by less than 0.3 kcal/mol are dropped as unrankable. But
rankability depends on noise: with r₁ = 0.815, the difference of two
single-replicate labels carries an uncertainty of about 0.61 of the between-design
spread, so many pairs above the threshold are still unrankable. Express the margin
as a multiple of the noise standard deviation and tune it within folds. Or replace
the hard cut with a soft weight: under a Gaussian noise model the probability that
the observed order matches the true order is Φ(|Δ|/σ_Δ), so weighting pairs by
2Φ − 1 keeps every pair while down-weighting the uncertain ones. The
five-replicate designs let you estimate σ_Δ per pair.

**Listwise losses instead of pairwise.** LambdaRank, LambdaLoss, ApproxNDCG and
NeuralNDCG weight pairs by their effect on the ranking metric rather than treating
all pairs alike, which suits noisy labels better.

**Retrain on pre-MPNN sequences.** The train/deploy mismatch is the largest
remaining inconsistency. Keeping the same MM/PBSA labels but training on the
pre-MPNN trajectory sequence matches the distributions. This is the single
cheapest improvement available.

**A pose-aware scorer.** MM/PBSA is a property of a complex; the current model
sees only sequences. That may be the ceiling on a sequence-only approach.

**Do not bother with a bigger model.** Six encoders spanning roughly 90-fold in
parameter count, masked-language-model pretraining on 376,011 in-distribution
sequences, and nine published checkpoints as initialisations all produced changes
smaller than the seed noise. Capacity is not the constraint.

### Training across two targets

The architecture already supports this, and it is the reason the target-side arm
was kept rather than collapsed into a bias.

The target encoder is frozen and its output cached, because within one campaign
the target never changes. But the arm still takes a target sequence as input, so
a model trained on data from several BCL-2 family members can distinguish them.

To extend to a second target, such as another BCL-2 family groove:

1. Generate and simulate designs against the new target, producing MM/PBSA labels
   in the same format.
2. Pool with the existing labels. Keep the target sequence as a column.
3. Unfreeze the target encoder, or cache one representation per target.
4. Stratify batches by target as well as by length, so that neither can be learned
   as a shortcut within a batch.
5. Hold out one target entirely as a test fold. This is the real question: does
   the model transfer to a groove it has never seen?

Because canonical grooves are structurally related across the family, a shared
model is plausible. It is also the natural route to counter-screening: a model
that scores a design against several grooves at once gives you a selectivity
signal rather than an affinity signal alone.

Worth knowing before you start: the designs here were never counter-screened, so
nothing in this pipeline has yet demonstrated selectivity.

---

## 4. Guided generation

Stage 1 again, with the affinity model attached to the discrete search.

### Configuration

`balm_config.json` holds everything specific to this fork.

```jsonc
{
  "balm_aux": {
    "enabled": true,
    "weight": 20.0,
    "direction": "minimize",
    "metric": "cosine",
    "active_stages": [4],
    "target_sequence": "<TARGET_SEQUENCE>",
    "checkpoint": "<CHECKPOINT_PATH>"
  },
  "structure_barrier": {
    "enabled": true,
    "weight": 100.0,
    "plddt_floor": 0.75,
    "iptm_floor": 0.75
  },
  "esm_naturalness": {
    "enabled": true,
    "weight": 1.0,
    "margin": 1.0
  }
}
```

### Run

```bash
conda activate <BINDCRAFT_ENV>
python bindcraft.py \
  --settings    settings_target/<target>.json \
  --filters     settings_filters/peptide_relaxed_filters.json \
  --advanced    settings_advanced/<protocol>.json \
  --balm_config balm_config.json
```

Check the coupling first with `test_balm_integration.py`.

### Why hard floors and not a weight

Weighting the affinity score against the native BindCraft objective does not
work. The search improves predicted affinity while driving binder pLDDT down to
roughly 0.35, into sequences AlphaFold2 no longer predicts will fold. No weight
avoids this: any weight large enough to steer the search is large enough to let
structure be traded away.

The floors change the accessible search space rather than rebalancing the
objective, so the affinity term can only move laterally among sequences that
remain foldable.

**The floors are attached independently of the affinity term.** Setting
`balm_aux.enabled: false` therefore gives a true unguided control: same loss
surface, same gates, same acceptance rules, differing only in the absence of the
score. Run one. Without it you cannot attribute any improvement to guidance.

### Settings worth changing

| Setting | Default | Note |
|---|---|---|
| `weight` | 20.0 | Scales the score's contribution. The floors do the structural work, so this is less delicate than it looks. |
| `direction` | `minimize` | **Checkpoint-dependent.** Some checkpoints score a stronger binder as a more negative cosine. Confirm which convention yours uses before running a campaign: an inverted setting steers away from strong binders while producing output that looks normal. |
| `active_stages` | `[4]` | Stage 4 only. Earlier stages are continuous, and the score is attached as a design post-callback that fires once per candidate. |
| `plddt_floor`, `iptm_floor` | 0.75 | Raising these reduces yield. Lowering them reinstates the collapse. |
| `target_sequence` | — | Must match the target the model was trained against. |

### Three further policies, active whenever guidance is on

- Stage 4 begins only if the stage-3 sequence already meets the pLDDT threshold.
- If the stage-4 winner fails the Cα clash or interface check, up to five
  runner-up candidates are tried.
- A trajectory is kept only if its final design came out of the guided search.

### Then close the loop

Simulate the new designs (stage 2), add them to the training pool, refit
(stage 3), and run again. Three passes were enough to show an effect here.

Be aware of what a longer loop risks. Each round refits a model on designs the
previous model scored well, then optimises new candidates against the refit. There
is no experimental measurement anywhere in it. MM/PBSA sits outside the trained
model, which keeps it honest for now, but MM/PBSA is itself a proxy with a
hand-calibrated dielectric and no entropy term. Further rounds could begin
selecting for its systematic weaknesses. A longer loop needs an external anchor:
more umbrella sampling, or better, experimental measurement.

---

## Analysis and reporting

| Question | Notebook |
|---|---|
| What predicts MM/PBSA? | `dSASA.ipynb`, `metric_corr_delta_total.ipynb` |
| Filter-passing versus rejected designs | `Rejected_accepted_analysis.ipynb` |
| Binding orientation assignment | `extract_n_to_c_orientation.ipynb` |
| Sequence logos, UMAP, between-set permutation test | `sequence_analysis.ipynb`, `new_seq.ipynb` |
| Guided versus control comparison | `Test_vs_control.ipynb`, `third_iter_analysis.ipynb` |

Two analysis points that matter for a new target:

**Assign binding orientation before any sequence analysis.** Designs bind in one
of two antiparallel orientations, which present opposite sequence registers to the
same surface. Pooling them yields a consensus describing neither. Every
sequence-level statistic must be computed within an orientation.

**Compute sequence-space statistics on the distance matrix, not on the UMAP.**
UMAP does not preserve distances. The projection is for display only.
