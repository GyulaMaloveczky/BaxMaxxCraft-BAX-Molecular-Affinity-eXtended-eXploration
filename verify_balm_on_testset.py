#!/usr/bin/env python3
"""
verify_balm_on_testset.py
=========================
RUN ON THE CLUSTER (checkpoint + GPU + CSV live there).

Scores every binder in the CSV with the BALM model your pipeline loads, then
correlates the BALM cosine against the ground-truth column **Y**.

Y = binding free energy  ==>  LOWER Y is a BETTER binder.

Therefore:
    cosine correlates NEGATIVELY with Y  -> higher cosine = better binder -> direction "maximize"
    cosine correlates POSITIVELY with Y  -> higher cosine = worse  binder -> direction "minimize"

Usage:
    python verify_balm_on_testset.py \
        --csv /FastHome/gyula/designs/place_holder/DATA/gy_14_more_iter/mpnn_design_stats_augmented.csv \
        --config balm_config.json [--seq-col Sequence] [--y-col Y] [--batch 16]
"""
from __future__ import annotations
import argparse, json, os, sys

DEFAULT_CSV = "/FastHome/gyula/designs/place_holder/DATA/gy_14_more_iter/mpnn_design_stats_augmented.csv"


def pick_seq_column(df):
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    cands = [c for c in df.columns if c.strip().lower() in ("sequence", "seq", "binder_sequence", "binder")]
    if not cands:
        cands = [c for c in df.columns if "seq" in c.strip().lower() and "id" not in c.strip().lower()]
    if not cands:
        raise SystemExit(f"No binder-sequence column found. Columns: {list(df.columns)}")
    cands.sort(key=lambda c: sum(set(str(v).upper()) <= aa and len(str(v)) >= 8
                                 for v in df[c].dropna().head(20)), reverse=True)
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "balm_config.json"))
    ap.add_argument("--seq-col", default=None)
    ap.add_argument("--y-col", default="Y")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="balm_vs_Y.png")
    ap.add_argument("--dump", default="balm_scores.csv")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import numpy as np, pandas as pd
    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception:
        def pearsonr(a, b):
            return float(np.corrcoef(a, b)[0, 1]), float("nan")
        def spearmanr(a, b):
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            return float(np.corrcoef(ra, rb)[0, 1]), float("nan")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import balm_model

    cfg = json.load(open(args.config))["balm_aux"]
    ckpt = args.checkpoint or cfg["checkpoint_path"]
    target = args.target or cfg["target_sequence"]
    device = args.device or cfg.get("device", "cuda")
    cfg_direction = cfg.get("direction")

    df = pd.read_csv(args.csv)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    if args.y_col not in df.columns:
        raise SystemExit(f"Y column {args.y_col!r} not in CSV. Columns: {list(df.columns)}")
    seq_col = args.seq_col or pick_seq_column(df)
    print(f"[load] checkpoint = {ckpt}")
    print(f"[load] config direction = {cfg_direction!r}")
    print(f"[data] rows={len(df)}  seq-col={seq_col!r}  y-col={args.y_col!r}")

    # clean sequences
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    seqs = df[seq_col].astype(str).str.strip().str.upper()
    ok = seqs.apply(lambda s: len(s) >= 1 and set(s) <= aa)
    if (~ok).any():
        print(f"[data] dropping {int((~ok).sum())} rows with non-AA/empty sequences")
    df, seqs = df[ok].reset_index(drop=True), seqs[ok].reset_index(drop=True)

    # score with the real model (batched + de-duped)
    model = balm_model.build_model(ckpt, device=device)
    import torch
    binders = seqs.tolist()
    uniq, todo = {}, sorted(set(binders))
    with torch.no_grad():
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            out = model(protein_seqs=[target] * len(chunk), proteina_seqs=chunk)
            for b, c in zip(chunk, out["cosine_similarity"].detach().cpu().numpy().reshape(-1)):
                uniq[b] = float(c)
            print(f"\r[score] {min(i + args.batch, len(todo))}/{len(todo)} unique binders", end="")
    print()
    df["balm_cosine"] = np.array([uniq[b] for b in binders])

    # correlate cosine vs Y
    x = df["balm_cosine"].to_numpy(float)
    y = pd.to_numeric(df[args.y_col], errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        raise SystemExit(f"Not enough usable (cosine, Y) pairs: n={n}")
    r, _ = pearsonr(x[m], y[m])
    rho, _ = spearmanr(x[m], y[m])

    # plot
    plt.figure(figsize=(6, 5))
    plt.scatter(x[m], y[m], s=14, alpha=0.5)
    b1, b0 = np.polyfit(x[m], y[m], 1)
    xs = np.linspace(x[m].min(), x[m].max(), 50)
    plt.plot(xs, b1 * xs + b0, "r-", lw=1.5)
    plt.xlabel("BALM cosine (prediction)")
    plt.ylabel(f"{args.y_col}  (binding free energy; lower = better)")
    plt.title(f"BALM cosine vs {args.y_col}\nPearson r={r:.3f}  Spearman rho={rho:.3f}  n={n}")
    plt.tight_layout(); plt.savefig(args.out, dpi=140)
    df.to_csv(args.dump, index=False)

    # verdict: lower Y = better. higher cosine better  <=>  r < 0
    higher_cos_is_better = r < 0
    should_be = "maximize" if higher_cos_is_better else "minimize"
    print("\n" + "=" * 70)
    print(f"Pearson r (cosine vs Y) = {r:+.3f}   Spearman rho = {rho:+.3f}   n = {n}")
    print(f"Y is binding free energy (lower = better binder).")
    print(f"=> higher BALM cosine corresponds to {'LOWER Y = STRONGER' if higher_cos_is_better else 'HIGHER Y = WEAKER'} binders")
    print(f"=> correct config is  direction = \"{should_be}\"")
    if cfg_direction and cfg_direction != should_be:
        print(f"   config says \"{cfg_direction}\"  --> BACKWARDS, flip it.")
    elif cfg_direction:
        print(f"   config says \"{cfg_direction}\"  --> consistent.")
    if abs(r) < 0.2:
        print("[caution] |r| < 0.2: BALM barely tracks Y on this set; direction call is weak.")
    print(f"[plot] {os.path.abspath(args.out)}")
    print(f"[dump] {os.path.abspath(args.dump)}")
    print("=" * 70)


if __name__ == "__main__":
    main()