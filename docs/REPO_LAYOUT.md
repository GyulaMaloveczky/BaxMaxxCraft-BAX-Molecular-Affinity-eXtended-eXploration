# BaxMaxxCraft Repository Layout

This repository is a research/computational design project with a mix of Python code, configuration, and generated outputs. The structure is organized to keep source code, configuration, and runtime artifacts cleanly separated.

## Source Code

- `bindcraft.py` — main entry point for binder design runs
- `bindcraft_augment.py` — adversarial/augmented sampling workflow
- `bindcraft_original.py` — upstream BindCraft file, retained for comparison
- `balm_*.py` — BAX affinity guidance modules:
  - `balm_model.py` — BALM-PPI model loading and inference
  - `balm_aux_loss.py` — loss adapter and scoring
  - `balm_design_callback.py` — stage-4 semi-greedy search with affinity
  - `balm_structure_guidance.py` — pLDDT, interface pTM, ESM2 floors
  - `balm_hydrophobic_guidance.py` — hydrophobic fraction capping
  - `balm_candidate_recorder.py` — per-candidate logging for analysis
  - `balm_colabdesign_bridge.py` — ColabDesign integration
  - `balm_jax_cache.py` — JAX caching utilities
  - `balm_profile.py` — performance profiling
  - `balm_animation.py` — trajectory visualization

## Core Libraries

- `functions/` — internal utilities and helpers:
  - `colabdesign_utils.py` — ColabDesign wrapper and four-stage protocol
  - `biopython_utils.py` — structural biology helpers
  - `generic_utils.py` — general-purpose utilities
  - `pyrosetta_utils.py` — PyRosetta integration
  - `augment_sampling.py` — adversarial sequence generation
  - `balm_jax_cache.py` — caching for JAX computations (also in root)

## Configuration

- `balm_config.json` — affinity guidance parameters (direction, weight, floors)
- `settings_target/` — target protein definitions (JSON)
- `settings_filters/` — filtering criteria panels (JSON)
- `settings_advanced/` — design protocol configurations (JSON)

## Testing & Validation

- `tests/` — test suite:
  - `test_balm_integration.py` — integration tests for affinity coupling
  - `verify_balm_on_testset.py` — checkpoint sanity checks

## Documentation & Assets

- `README.md` — project overview and setup
- `BALM_AUX_README.md` — detailed BALM-AUX loss documentation
- `TODO.md` — outstanding tasks
- `figures/` — paper figures and diagrams
- `notebooks/` — Colab notebooks
- `docs/` — additional documentation

## Generated Outputs (Git-ignored)

- `logs/` — runtime logs and stdout/stderr
- `outputs/` — designed binder sequences and structures
- `artifacts/` — model checkpoints and datasets
- `slurm_output_*/` — cluster job logs
- `params/` — local parameter files

## External Dependencies

- `functions/DAlphaBall.gcc` — compiled binary (consider Git LFS)
- `functions/dssp` — compiled DSSP tool (consider Git LFS)
- `install_bindcraft.sh` — installation script for dependencies

## Notes

The repository follows a clean-source philosophy:
- Source code and config live in version control
- Runtime outputs, logs, and artifacts are generated locally and ignored
- Test code is separated into a dedicated directory
- Large binaries should eventually migrate to Git LFS or separate release artifacts
