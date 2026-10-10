#!/usr/bin/env bash
set -euo pipefail

# Example wrapper for a standard design run.
# Update the target settings JSON to match your desired design campaign.

python bindcraft.py \
  --settings settings_target/<your_target>.json \
  --filters settings_filters/peptide_relaxed_filters.json \
  --advanced settings_advanced/peptide_4stage_multimer.json \
  --balm_config balm_config.json
