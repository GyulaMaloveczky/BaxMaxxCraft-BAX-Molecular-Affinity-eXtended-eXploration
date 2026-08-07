# BALM-PPI auxiliary loss — integration guide

Plugs the fine-tuned BALM-PPI affinity model into your multi-stage design pipeline
as an extra loss term, active only in the **discrete** stages (3 and 4).

## Files
- `balm_model.py` — the BALM-PPI architecture + checkpoint loader (auto-detects LoRA
  rank / target modules / projection size, so any fold loads as-is). Shared with the
  training notebook.
- `balm_aux_loss.py` — the adapter: JSON-driven, stage-gated, cached, with self-tests.
- `design_config.balm_aux.json` — the config block to merge into your pipeline's JSON.

## Wire-in (three lines in the host code)
```python
from balm_aux_loss import BalmAuxLoss

# once, at setup — cfg is your parsed JSON:
balm_aux = BalmAuxLoss.from_config(cfg["balm_aux"])
balm_aux.run_sanity_checks()                       # fail fast on misconfig

# inside the per-candidate evaluation of each stage:
total_loss += balm_aux.loss(current_sequence, stage=stage_index)
```
`loss(...)` returns `0.0` on any non-discrete stage, when `enabled: false`, or when
`weight: 0`, so calling it from every stage is safe.

## The `weight` key
`weight` in the JSON scales the loss contribution. With `direction: "maximize"`,
`loss = -weight * score`, so higher `weight` pushes the search harder toward strong
predicted binders. Start at `1.0` with `metric: "cosine"` (bounded [-1, 1], easy to
balance against your other losses) and adjust.

## Assumptions I had to make (the host pipeline wasn't in the upload)
The file you attached was `MobaXterm.exe` (the SSH client), not the pipeline source,
so this is built against a documented interface. Check these against your code:

1. **Stages 3 & 4 are gradient-free** (mutation/accept-reject on the loss value).
   The BALM model is called as a pure scorer — no autodiff bridge — so it works even
   if the host is JAX. If a stage you target actually needs gradients w.r.t. the
   sequence, this hook won't propagate them (tell me and I'll add a bridge).
2. **"Current sequence" = the binder string** being designed; the **target is fixed**
   and read from `target_sequence` in the JSON. If your target varies per candidate,
   pass it explicitly: `balm_aux.loss(seq, stage, target=...)`.
3. **Stage indexing is an int** matching `active_stages`. If your stages are named
   ("hard", "semigreedy", ...) map them to ints, or tell me the names and I'll switch
   the gating to strings.
4. **Lower total loss = better** in the host. `direction: "maximize"` assumes stronger
   binding should lower the loss; flip to `"minimize"` if your convention is inverted.

## Sanity checks
`run_sanity_checks()` verifies: config validity, stage gating, `weight: 0` no-op,
checkpoint loads, finite in-range score, determinism, caching, correct loss sign, and
linear weight scaling. Run offline first without touching the GPU/checkpoint:
```bash
python balm_aux_loss.py design_config.balm_aux.json --no-model
```
(Full config/logic layer is already verified with a mock; the model-load checks need
the real checkpoint + torch on the cluster.)
