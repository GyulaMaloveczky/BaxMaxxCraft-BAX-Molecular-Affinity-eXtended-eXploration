"""test_balm_integration.py
========================
Checks that BALM is ACTUALLY wired into BindCraft, at three depths:

  python test_balm_integration.py                  # offline: no jax / no torch
  python test_balm_integration.py --jax            # + exercise the jitted loss_fn (needs jax)
  python test_balm_integration.py --model CONFIG    # + load the real checkpoint (needs torch+GPU)

Offline tests use a call-counting mock predictor, so they prove the wiring logic
(attachment, gating, weighting, sequence decode, host-callback firing) without the
650M model. --model runs the real BalmAuxLoss.run_sanity_checks(load_model=True).
"""
import sys
import numpy as np

from balm_aux_loss import BalmAuxLoss, BalmAuxConfig
from balm_colabdesign_bridge import add_balm_loss, _indices_to_sequence, RESTYPES


# --------------------------------------------------------------------------- #
# Test doubles                                                                 #
# --------------------------------------------------------------------------- #
class MockPredictor:
    """Deterministic pseudo-cosine from the binder; counts real (uncached) calls."""
    def __init__(self):
        self.calls = 0
    def score(self, target, binder):
        self.calls += 1
        h = (sum(ord(c) for c in binder) % 1000) / 500.0 - 1.0
        return max(-0.999, min(0.999, h))


class StubAfModel:
    """Minimal stand-in for a ColabDesign af_model: just what add_balm_loss touches."""
    def __init__(self, binder_len=40):
        self._callbacks = {"model": {"loss": []}}
        self.opt = {"weights": {}}
        self._binder_len = binder_len


def _mk_aux(weight=1.0, stages=(4,), predictor=None):
    cfg = {
        "enabled": True, "weight": weight,
        "checkpoint_path": "/not/used/mock.pth",
        "target_sequence": "MTARGETSEQUENCE",
        "active_stages": list(stages), "metric": "cosine",
        "direction": "maximize", "device": "cpu",
    }
    return BalmAuxLoss.from_config(cfg, predictor=predictor or MockPredictor())


def _ok(name):
    print(f"  [PASS] {name}")


# --------------------------------------------------------------------------- #
# Offline tests (no jax, no torch)                                             #
# --------------------------------------------------------------------------- #
def test_decode():
    assert _indices_to_sequence([0, 1, 2, 3, 4, 19, 18, 17]) == "ARNDCVYW"
    assert _indices_to_sequence([0, 99, 1]) == "AXR"
    assert len(RESTYPES) == 20
    _ok("residue decode + RESTYPES length")


def test_attach_registers_weight_and_callback():
    aux = _mk_aux(weight=0.7)
    model = StubAfModel(binder_len=40)
    n0 = len(model._callbacks["model"]["loss"])
    fn = add_balm_loss(model, aux, weight=0.7)
    assert model.opt["weights"]["balm"] == 0.7, "weight not registered"
    assert len(model._callbacks["model"]["loss"]) == n0 + 1, "loss callback not appended"
    assert callable(fn) and fn is model._callbacks["model"]["loss"][-1]
    assert hasattr(fn, "host_penalty")
    _ok("add_balm_loss registers opt['weights']['balm'] + appends one loss callback")


def test_host_penalty_sign_and_binder_slice():
    pred = MockPredictor()
    aux = _mk_aux(weight=1.0, predictor=pred)
    model = StubAfModel(binder_len=10)
    fn = add_balm_loss(model, aux, weight=1.0)

    # full-complex indices (target+binder); host must score only the last binder_len.
    full_idx = np.array([0] * 25 + list(range(10)))   # 25 target + 10 binder residues
    penalty = fn.host_penalty(full_idx)
    binder_str = _indices_to_sequence(full_idx[-10:])
    expected = -pred.score("MTARGETSEQUENCE", binder_str)
    assert np.isclose(float(penalty), expected), (penalty, expected)
    assert pred.calls >= 1
    _ok("host_penalty scores binder-only (last binder_len) and returns -cosine")


def test_gating_and_zero_weight():
    aux = _mk_aux(weight=2.0, stages=(4,))
    # loss() from the adapter (used for the MPNN filter path) must gate by stage/weight
    assert aux.loss("ACDEFGHIKL", stage=3) == 0.0, "should be inactive on stage 3"
    assert aux.loss("ACDEFGHIKL", stage=4) != 0.0, "should be active on stage 4"
    aux.cfg.weight = 0.0
    assert aux.loss("ACDEFGHIKL", stage=4) == 0.0, "zero weight must be a no-op"
    _ok("stage gating + zero-weight no-op")


def test_direction_sign():
    # Stage-4 penalty must follow cfg.direction:
    #   maximize -> penalty = -cos (drive cosine up)
    #   minimize -> penalty = +cos (drive cosine down / more negative = better)
    for direction, expect_sign in [("maximize", -1.0), ("minimize", +1.0)]:
        aux = _mk_aux(predictor=MockPredictor())
        aux.cfg.direction = direction
        model = StubAfModel(binder_len=12)
        fn = add_balm_loss(model, aux, weight=1.0)
        cos = aux.raw_score(_indices_to_sequence(np.arange(12)))
        penalty = float(fn.host_penalty(np.arange(12)))
        assert np.isclose(penalty, expect_sign * cos), (direction, penalty, cos)
    _ok("host_penalty respects cfg.direction (maximize=-cos, minimize=+cos)")


def test_passes_filter_direction():
    aux = _mk_aux(predictor=MockPredictor()); aux.cfg.direction = "minimize"
    aux.cfg.min_score = None
    assert aux.passes("ACDEFGHIKL") is True, "min_score=None must keep everything"
    b = "ACDEFGHIKL"; s = aux.raw_score(b)
    aux.cfg.min_score = s + 0.1   # minimize: keep score <= threshold
    assert aux.passes(b) is True
    aux.cfg.min_score = s - 0.1
    assert aux.passes(b) is False
    _ok("passes() gate is direction-aware")


def test_active_flag_skips_esm():
    pred = MockPredictor()
    aux = _mk_aux(predictor=pred)
    model = StubAfModel(binder_len=8)
    flag = {"on": False}
    fn = add_balm_loss(model, aux, weight=1.0, active_flag=flag)
    fn.host_penalty(np.arange(8))
    assert pred.calls == 0, "predictor should NOT run when active_flag off"
    flag["on"] = True
    fn.host_penalty(np.arange(8))
    assert pred.calls == 1, "predictor should run when active_flag on"
    _ok("active_flag gates the ESM forward")


# --------------------------------------------------------------------------- #
# --jax : exercise the real jitted loss_fn end-to-end with the mock predictor  #
# --------------------------------------------------------------------------- #
def test_loss_fn_fires_under_jax():
    import jax
    import jax.numpy as jnp
    pred = MockPredictor()
    aux = _mk_aux(weight=1.5, predictor=pred)
    model = StubAfModel(binder_len=12)
    fn = add_balm_loss(model, aux, weight=1.5)

    L = 12
    one_hot = np.zeros((1, L, 20), dtype=np.float32)      # (num_seq, L, 20)
    residues = np.array([i % 20 for i in range(L)])
    one_hot[0, np.arange(L), residues] = 1.0
    inputs = {"seq": jnp.asarray(one_hot)}

    calls_before = pred.calls
    out = jax.jit(lambda inp: fn(inp, {}))(inputs)        # jit to prove pure_callback path
    val = float(out["balm"])
    assert "balm" in out and np.isfinite(val), out
    assert pred.calls == calls_before + 1, "host callback did not fire under jit"
    expected = -pred.score("x", _indices_to_sequence(residues))
    # (mock ignores target; compare on the same binder)
    assert np.isclose(val, expected, atol=1e-5), (val, expected)
    _ok("loss_fn returns finite {'balm': ...} and pure_callback fires under jax.jit")


# --------------------------------------------------------------------------- #
# --model : load the real checkpoint                                           #
# --------------------------------------------------------------------------- #
def test_real_model(config_path):
    aux = BalmAuxLoss.from_json(config_path)
    aux.run_sanity_checks(load_model=True)                 # loads fold checkpoint, checks scores
    s1 = aux.raw_score("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
    s2 = aux.raw_score("GSHMKLVFFAEDVGSNKGAIIGLMVGGVVIA")
    assert -1.0001 <= s1 <= 1.0001 and -1.0001 <= s2 <= 1.0001
    print(f"  [PASS] real checkpoint scores: {s1:.4f}, {s2:.4f}; stats={aux.stats()}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Offline wiring tests:")
    test_decode()
    test_attach_registers_weight_and_callback()
    test_host_penalty_sign_and_binder_slice()
    test_gating_and_zero_weight()
    test_direction_sign()
    test_passes_filter_direction()
    test_active_flag_skips_esm()

    if "--jax" in sys.argv:
        print("JAX loss_fn test:")
        test_loss_fn_fires_under_jax()

    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        cfg = sys.argv[i + 1] if i + 1 < len(sys.argv) else "balm_config.json"
        print(f"Real-model test ({cfg}):")
        test_real_model(cfg)

    print("\nAll requested tests passed.")
