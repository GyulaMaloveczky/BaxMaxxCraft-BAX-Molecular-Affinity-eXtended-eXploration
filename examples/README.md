# Examples and cookbook snippets

This folder is for short usage examples, configuration snippets, and minimal run recipes.

## Standard workflow

```bash
python bindcraft.py \
  --settings settings_target/<your_target>.json \
  --filters settings_filters/peptide_relaxed_filters.json \
  --advanced settings_advanced/peptide_4stage_multimer.json \
  --balm_config balm_config.json
```

## Validation workflow

```bash
python tests/test_balm_integration.py
python tests/verify_balm_on_testset.py --config balm_config.json --batch 16
```
