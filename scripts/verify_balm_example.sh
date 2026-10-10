#!/usr/bin/env bash
set -euo pipefail

# Example wrapper for BALM sanity checks.

python tests/verify_balm_on_testset.py \
  --csv /path/to/design_stats.csv \
  --config balm_config.json \
  --batch 16
