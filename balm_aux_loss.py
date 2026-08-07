"""
balm_aux_loss.py
================
Auxiliary affinity loss that plugs a fine-tuned BALM-PPI model into a
multi-stage protein-design pipeline.

WHAT IT DOES
------------
During the DISCRETE stages of the host pipeline (stages 3 and 4 by default), the
pipeline evaluates a concrete candidate sequence. This adapter scores that
sequence with a fine-tuned BALM-PPI model and returns a loss contribution that
REWARDS stronger predicted binding. Because the discrete stages accept/reject on
the loss *value* (no gradient flows through the sequence), the PyTorch BALM model
is used purely as a scoring function -- there is no JAX<->PyTorch autodiff bridge
to build, and the host may be in any framework.

Everything is driven from the pipeline's JSON config (see EXAMPLE_CONFIG below).
The `weight` key controls the strength of the new loss.

WIRING (the only change you make in the host code)
--------------------------------------------------
    from balm_aux_loss import BalmAuxLoss

    # once, at setup (cfg is your parsed JSON dict):
    balm_aux = BalmAuxLoss.from_config(cfg["balm_aux"])
    balm_aux.run_sanity_checks()          # fail fast if anything is misconfigured

    # inside the per-candidate evaluation of a discrete stage:
    total_loss += balm_aux.loss(current_sequence, stage=stage_index)

`loss(...)` returns exactly 0.0 when the stage is not discrete, when disabled, or
when weight == 0, so it is safe to call from every stage unconditionally.

The heavy model is only imported/loaded when a real predictor is first needed, so
importing this module (and running the mock-based sanity checks) requires no torch.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


EXAMPLE_CONFIG = {
    "balm_aux": {
        "enabled": True,
        "weight": 1.0,                       # <-- strength of the new loss (from JSON)
        "checkpoint_path": "/FastHome/gyula/repos/BALM-PPI/BALM-PPI/results/balm_ppi_phi/best_model_fold_4.pth",
        "target_sequence": "REPLACE_WITH_TARGET_PROTEIN_SEQUENCE",
        "active_stages": [3, 4],             # discrete stages only
        "metric": "cosine",                  # "cosine" (bounded [-1,1]) or "affinity"
        "affinity_bounds": [25.0, 55.0],     # used only when metric == "affinity"
        "direction": "maximize",             # reward stronger binding
        "device": "cuda",
        "fail_open": False,                  # on predict error: raise (False) or skip w/ warning (True)
        "cache_size": 4096,
    }
}


# --------------------------------------------------------------------------- #
# Predictor: wraps the BALM model and exposes .score(target, binder) -> float  #
# --------------------------------------------------------------------------- #
class BalmPredictor:
    """Lazy-loading wrapper around a BALM-PPI checkpoint.

    Exposes `.score(target, binder) -> float` (raw cosine similarity in [-1, 1]).
    Kept separate from the loss so it can be mocked/injected for offline tests.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None
        # target seq -> cached L2-normalized PROJECTED target vector.
        # Caches the whole target path: the ESM mean-pooled embedding AND
        # projection_head.protein_projection + normalize applied to it.
        self._target_proj = {}

    def _ensure_model(self):
        if self._model is None:
            import balm_model  # local import: only needs torch when actually used
            self._model = balm_model.build_model(self.checkpoint_path, device=self.device)
        return self._model

    def _project_target(self, model, target: str):
        """L2-normalized projected target vector. Computed once per target sequence."""
        import torch
        import torch.nn.functional as F
        cached = self._target_proj.get(target)
        if cached is not None:
            return cached
        head = model.projection_head
        with torch.no_grad():
            t_emb = model._embed([target])                     # ESM pass: ONCE per target
            t_proj = F.normalize(
                head.protein_projection(head.dropout(t_emb)),  # projection: ONCE per target
                p=2, dim=1,
            )
        self._target_proj[target] = t_proj
        return t_proj

    def score(self, target: str, binder: str) -> float:
        import torch
        import torch.nn.functional as F
        model = self._ensure_model()
        # Caching the target's post-dropout projection is only sound in eval mode;
        # in train mode the target would keep one frozen dropout mask while the
        # binder resampled a fresh one, silently biasing every score.
        if model.training:
            raise RuntimeError(
                "[balm_aux] BALM model is in train mode; the cached target projection "
                "would freeze a dropout mask. Call model.eval() before scoring."
            )
        head = model.projection_head
        t_proj = self._project_target(model, target)           # target arm: fully cached
        with torch.no_grad():
            b_emb = model._embed([binder])                     # binder arm: only this per call
            b_proj = F.normalize(
                head.proteina_projection(head.dropout(b_emb)), p=2, dim=1
            )
            cos = torch.clamp(F.cosine_similarity(t_proj, b_proj), -0.9999, 0.9999)
        return float(cos.item())


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class BalmAuxConfig:
    enabled: bool = True
    weight: float = 1.0
    checkpoint_path: str = ""
    target_sequence: str = ""
    active_stages: frozenset = field(default_factory=lambda: frozenset({3, 4}))
    metric: str = "cosine"                    # "cosine" | "affinity"
    affinity_bounds: tuple = (25.0, 55.0)
    direction: str = "maximize"               # "maximize" | "minimize"
    device: str = "cuda"
    fail_open: bool = False
    cache_size: int = 4096
    min_score: Optional[float] = None   # optional MPNN-filter threshold (cosine units)

    @staticmethod
    def from_dict(d: dict) -> "BalmAuxConfig":
        d = dict(d)
        _min = d.get("min_score", d.get("balm_min_score"))
        cfg = BalmAuxConfig(
            enabled=bool(d.get("enabled", True)),
            weight=float(d.get("weight", 1.0)),
            checkpoint_path=str(d.get("checkpoint_path", "")),
            target_sequence=str(d.get("target_sequence", "")),
            active_stages=frozenset(int(s) for s in d.get("active_stages", [3, 4])),
            metric=str(d.get("metric", "cosine")).lower(),
            affinity_bounds=tuple(d.get("affinity_bounds", (25.0, 55.0))),
            direction=str(d.get("direction", "maximize")).lower(),
            device=str(d.get("device", "cuda")),
            fail_open=bool(d.get("fail_open", False)),
            cache_size=int(d.get("cache_size", 4096)),
            min_score=(None if _min is None else float(_min)),
        )
        cfg.validate()
        return cfg

    def validate(self):
        errors = []
        if self.metric not in ("cosine", "affinity"):
            errors.append(f"metric must be 'cosine' or 'affinity', got {self.metric!r}")
        if self.direction not in ("maximize", "minimize"):
            errors.append(f"direction must be 'maximize' or 'minimize', got {self.direction!r}")
        if not isinstance(self.weight, (int, float)) or not math.isfinite(self.weight):
            errors.append(f"weight must be a finite number, got {self.weight!r}")
        if not self.active_stages:
            errors.append("active_stages is empty; nothing would ever fire")
        if self.metric == "affinity":
            lo, hi = self.affinity_bounds
            if not (hi > lo):
                errors.append(f"affinity_bounds must be (lo, hi) with hi>lo, got {self.affinity_bounds}")
        if self.enabled and not self.checkpoint_path:
            errors.append("checkpoint_path is empty but the aux loss is enabled")
        if self.enabled and not self.target_sequence:
            errors.append("target_sequence is empty but the aux loss is enabled")
        if errors:
            raise ValueError("Invalid balm_aux config:\n  - " + "\n  - ".join(errors))


# --------------------------------------------------------------------------- #
# The loss adapter                                                             #
# --------------------------------------------------------------------------- #
class BalmAuxLoss:
    def __init__(self, config: BalmAuxConfig, predictor: Optional[object] = None):
        self.cfg = config
        # predictor may be injected (e.g. a mock for tests); otherwise built lazily.
        self._predictor = predictor
        self._cache: dict = {}
        self._n_calls = 0
        self._n_cache_hits = 0
        self._n_predict = 0

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_config(cls, cfg_dict: dict, predictor: Optional[object] = None) -> "BalmAuxLoss":
        return cls(BalmAuxConfig.from_dict(cfg_dict), predictor=predictor)

    @classmethod
    def from_json(cls, path: str, key: str = "balm_aux", predictor=None) -> "BalmAuxLoss":
        with open(path) as f:
            cfg = json.load(f)
        return cls.from_config(cfg[key], predictor=predictor)

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = BalmPredictor(self.cfg.checkpoint_path, self.cfg.device)
        return self._predictor

    # -- gating ------------------------------------------------------------- #
    def is_active(self, stage: int) -> bool:
        return (
            self.cfg.enabled
            and self.cfg.weight != 0.0
            and int(stage) in self.cfg.active_stages
        )

    # -- scoring ------------------------------------------------------------ #
    def raw_score(self, binder: str, target: Optional[str] = None) -> float:
        """Cosine similarity in [-1, 1] for (target, binder). Cached."""
        tgt = target if target is not None else self.cfg.target_sequence
        key = (tgt, binder)
        if key in self._cache:
            self._n_cache_hits += 1
            return self._cache[key]
        self._n_predict += 1
        score = float(self.predictor.score(tgt, binder))
        if len(self._cache) < self.cfg.cache_size:
            self._cache[key] = score
        return score

    def _metric_value(self, cos: float) -> float:
        if self.cfg.metric == "affinity":
            import balm_model
            return balm_model.cos_to_affinity(cos, *self.cfg.affinity_bounds)
        return cos

    def loss(self, binder: str, stage: int, target: Optional[str] = None) -> float:
        """Loss contribution to ADD to the host's total loss.

        Returns 0.0 when the stage is not active (not discrete / disabled / weight 0),
        so it is safe to call unconditionally from every stage. Stronger predicted
        binding => more-negative contribution (when direction == 'maximize').
        """
        self._n_calls += 1
        if not self.is_active(stage):
            return 0.0
        try:
            cos = self.raw_score(binder, target=target)
        except Exception as e:  # noqa: BLE001
            if self.cfg.fail_open:
                warnings.warn(f"[balm_aux] prediction failed, skipping (fail_open): {e}")
                return 0.0
            raise
        value = self._metric_value(cos)
        sign = -1.0 if self.cfg.direction == "maximize" else +1.0
        contribution = sign * self.cfg.weight * value
        if not math.isfinite(contribution):
            raise ValueError(f"[balm_aux] non-finite loss contribution: {contribution}")
        return float(contribution)

    def passes(self, binder: str, target: Optional[str] = None) -> bool:
        """Direction-aware MPNN filter gate. True = keep the sequence.

        With min_score = None, always keeps. With direction 'minimize' (more negative
        cosine = better), keeps sequences whose score is <= min_score; with 'maximize',
        keeps score >= min_score.
        """
        if self.cfg.min_score is None:
            return True
        s = self.raw_score(binder, target=target)
        return s <= self.cfg.min_score if self.cfg.direction == "minimize" else s >= self.cfg.min_score

    def stats(self) -> dict:
        return {
            "calls": self._n_calls,
            "predictions": self._n_predict,
            "cache_hits": self._n_cache_hits,
            "cache_size": len(self._cache),
        }

    # -- self-tests --------------------------------------------------------- #
    def run_sanity_checks(self, probe_binders: Optional[Iterable[str]] = None,
                          load_model: bool = True, verbose: bool = True) -> dict:
        """Verify the adapter is wired correctly. Raises on hard failures.

        Set load_model=False to check only the config/gating/weight logic without
        touching torch or the checkpoint (uses whatever predictor is injected).
        """
        results = {}
        probe_binders = list(probe_binders or ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
                                               "GSHMKLVFFAEDVGSNKGAIIGLMVGGVVIA"])

        def check(name, cond, detail=""):
            results[name] = bool(cond)
            if verbose:
                print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
                      + (f" -- {detail}" if detail and not cond else ""))
            if not cond:
                raise AssertionError(f"balm_aux sanity check failed: {name}. {detail}")

        if verbose:
            print("Running balm_aux sanity checks...")

        # 1. config is internally valid
        self.cfg.validate()
        check("config_valid", True)

        # 2. gating: an inactive (non-discrete) stage must contribute nothing
        inactive_stage = next((s for s in range(0, 20) if s not in self.cfg.active_stages), -1)
        check("gating_inactive_stage_is_zero",
              self.loss(probe_binders[0], stage=inactive_stage) == 0.0,
              f"stage {inactive_stage} should be inactive")

        # 3. weight == 0 must be a no-op even on an active stage
        saved_w = self.cfg.weight
        self.cfg.weight = 0.0
        active_stage = min(self.cfg.active_stages)
        check("zero_weight_is_zero",
              self.loss(probe_binders[0], stage=active_stage) == 0.0)
        self.cfg.weight = saved_w

        if not load_model and self._predictor is None:
            if verbose:
                print("  (skipping model-dependent checks: load_model=False, no predictor)")
            return results

        # 4. checkpoint loads / predictor produces a finite scalar
        cos = self.raw_score(probe_binders[0])
        check("score_is_finite", math.isfinite(cos), f"got {cos}")

        # 5. cosine is in range
        check("cosine_in_range", -1.0 - 1e-4 <= cos <= 1.0 + 1e-4, f"got {cos}")

        # 6. determinism (eval mode, cache): same input -> same score
        check("deterministic", self.raw_score(probe_binders[0]) == cos)

        # 7. cache actually served the repeat call
        check("cache_hit_registered", self._n_cache_hits >= 1)

        # 8. active-stage loss is finite and has the correct sign vs. score
        l = self.loss(probe_binders[0], stage=active_stage)
        expected_sign = -1.0 if self.cfg.direction == "maximize" else 1.0
        val = self._metric_value(cos)
        check("active_loss_finite", math.isfinite(l))
        if val != 0:
            check("loss_sign_matches_direction",
                  (l == 0.0) or (math.copysign(1.0, l) == math.copysign(1.0, expected_sign * val)),
                  f"loss={l}, value={val}, direction={self.cfg.direction}")

        # 9. monotonicity: doubling weight scales the contribution
        self.cfg.weight = saved_w * 2
        l2 = self.loss(probe_binders[0], stage=active_stage)
        self.cfg.weight = saved_w
        check("weight_scales_linearly", math.isclose(l2, 2 * l, rel_tol=1e-6, abs_tol=1e-9))

        if verbose:
            print(f"All checks passed. stats={self.stats()}")
        return results


# --------------------------------------------------------------------------- #
# CLI: `python balm_aux_loss.py config.json` -> validate + (optional) load     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python balm_aux_loss.py <config.json> [--no-model]")
        print("\nExample config:\n" + json.dumps(EXAMPLE_CONFIG, indent=2))
        sys.exit(0)
    aux = BalmAuxLoss.from_json(sys.argv[1])
    aux.run_sanity_checks(load_model="--no-model" not in sys.argv)
