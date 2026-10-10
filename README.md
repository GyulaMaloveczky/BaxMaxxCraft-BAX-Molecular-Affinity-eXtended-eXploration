# BaxMaxxCraft repository structure

This repository is organized into a few clearly separated areas:

- `docs/` — project and repo-level documentation
- `tests/` — validation and sanity-check scripts
- `scripts/` — runnable helper scripts for common workflows
- `examples/` — example usage and config snippets
- `reference/` — upstream or comparison files that are kept for context
- `logs/` and `outputs/` — generated artifacts (ignored by Git)

## Most important files

- `bindcraft.py` — main design entry point
- `balm_config.json` — affinity-model configuration
- `balm_*.py` — affinity guidance modules
- `functions/` — pipeline utilities and helpers
- `settings_*/` — design settings for target, filter, and advanced protocols

## Working conventions

- Keep model code and pipeline logic in the root or `functions/`
- Keep validation scripts under `tests/`
- Keep helper scripts under `scripts/`
- Keep raw run logs and generated outputs out of version control
- Keep upstream or comparison files under `reference/` when they are not part of the live path
