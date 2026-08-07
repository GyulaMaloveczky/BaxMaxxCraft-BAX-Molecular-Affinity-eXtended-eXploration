"""
augment_sampling.py
===================
"Augment sampling" mode: mine a BindCraft trajectory for TRAINING DATA instead of
for binders.

A normal trajectory throws away almost everything it computes. Stage 4 evaluates
`greedy_iterations * tries` sequences with a full AF2 pass each, keeps the argmin
of the loss, and discards the rest (af/design.py:455-465). This mode keeps all of
it:

  Augmentation.csv   one row per sequence EVALUATED during semigreedy, with the
                     scores that evaluation already produced. Nothing here is
                     recomputed -- it is the loss dict AF2 handed back, plus the
                     BALM score, plus which mutation produced the candidate.

  Bad_examples.csv   one row per selected candidate that survived to MPNN and
                     then failed the filters. These are the interesting ones:
                     BALM liked the sequence, the pipeline did not.

WHAT COUNTS AS A BAD EXAMPLE
    A candidate is bad if any filter it failed is NOT in `ignore`, and `ignore`
    is hydrophobicity only. Stage 4 already has explicit barrier losses for it
    (add_hydrophobic_fraction_barrier_loss, add_interface_hydrophobic_barrier_loss),
    so failing it means a guard leaked, not that BALM was wrong. Everything else
    counts -- clashes and interface residue count included.

    Note those last two are bypassed as TRAJECTORY-level gates on the way in
    (augment_ignore_trajectory_verdict), so a clashing trajectory still reaches
    MPNN; if its redesigns then trip the clash filter, that is a real failure and
    is recorded. The gate and the filter are different checks on different poses.

    To invert the rule -- collect only what fails the ignored filters -- set
    advanced_settings["augment_bad_example_mode"] = "only". To change which
    filters are ignored, set advanced_settings["augment_ignored_filters"].

    A candidate whose MPNN redesigns are evaluated is judged on the BEST of them:
    if ANY MPNN sequence passes every filter, the candidate is a pass and is not
    written to Bad_examples.csv at all.

SELECTION (which candidates go on to MPNN)
    Sort the deduplicated candidates by BALM score, ascending (BALM is a distance
    /energy-like score, so lower = better; that is the same direction the design
    callback minimises).
      * best score >= 0  ->  pass ONE candidate, the best.
      * best score <  0  ->  pass every candidate with a negative score, most
                             negative first, capped at `max_negative` (5).

COST
    Recording is free when BALM is on -- the design callback has already scored
    every candidate and raw_score() is cached. With BALM off, scoring each
    candidate for the CSV costs one ESM pass per candidate; set
    advanced_settings["augment_score_balm"] = false to skip it (the BALM columns
    then come out empty and selection falls back to loss order).
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

# Canonical residue order -- matches "".join(residue_constants.restypes) and the
# _RESTYPES in colabdesign_utils.py. Kept local so this module imports cleanly
# without jax/colabdesign present (it is unit-testable on its own).
RESTYPES = "ARNDCQEGHILKMFPSTWYV"

# Filter columns that do NOT make a candidate a bad example: hydrophobicity, and
# only hydrophobicity. Stage 4 already walls it off explicitly
# (add_hydrophobic_fraction_barrier_loss / add_interface_hydrophobic_barrier_loss),
# so a design failing it failed a guard rather than telling us anything new.
# Everything else counts -- including clashes and interface residue count, which
# ARE recorded as failures even though the trajectory-level versions of those
# checks are bypassed on the way in (see augment_ignore_trajectory_verdict).
DEFAULT_IGNORED_FILTERS = (
    "Surface_Hydrophobicity",
    "Interface_Hydrophobicity",
)

# check_filters() returns column names that may carry a per-model prefix; the
# same list bindcraft.py uses when incrementing failure counts.
_FILTER_PREFIXES = ("Average_", "1_", "2_", "3_", "4_", "5_")

_CSV_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# settings                                                                     #
# --------------------------------------------------------------------------- #
def augment_enabled(advanced_settings):
    return bool(advanced_settings.get("augment_sampling", False))


def augment_settings(advanced_settings):
    """Everything this mode reads, with defaults, in one place."""
    return {
        "enabled": augment_enabled(advanced_settings),
        "max_negative": int(advanced_settings.get("augment_max_negative", 5)),
        "score_balm": bool(advanced_settings.get("augment_score_balm", True)),
        "dedupe": bool(advanced_settings.get("augment_dedupe_sequences", False)),
        "mode": str(advanced_settings.get("augment_bad_example_mode", "exclude")),
        "ignored": tuple(advanced_settings.get("augment_ignored_filters",
                                               DEFAULT_IGNORED_FILTERS)),
        # run MPNN on selected candidates even if the trajectory itself would
        # normally be rejected (clashing / low confidence). The whole point is to
        # collect failures, so this defaults to True.
        "ignore_trajectory_verdict": bool(
            advanced_settings.get("augment_ignore_trajectory_verdict", True)),
    }


def augment_paths(design_path):
    return {
        "augmentation": os.path.join(design_path, "Augmentation.csv"),
        "bad_examples": os.path.join(design_path, "Bad_examples.csv"),
    }


# --------------------------------------------------------------------------- #
# records                                                                      #
# --------------------------------------------------------------------------- #
class CandidateRecord:
    """One sequence evaluated during semigreedy."""

    __slots__ = ("k", "iteration", "try_index", "sequence", "scores", "balm",
                 "mutation", "source", "accepted", "seen")

    def __init__(self, k, iteration, try_index, sequence, scores, balm=None,
                 mutation=None, source="mutation"):
        self.k = k
        self.iteration = iteration
        self.try_index = try_index
        self.sequence = sequence
        self.scores = scores or {}
        self.balm = balm
        self.mutation = mutation          # (pos, aa_letter) or None
        self.source = source              # "mutation" | "baseline" | "replay"
        self.accepted = False             # semigreedy kept this one
        self.seen = 1                     # times this exact sequence was evaluated

    @property
    def loss(self):
        v = self.scores.get("loss")
        return float(v) if v is not None else float("inf")

    def row(self, **extra):
        r = {"iteration": self.iteration, "try": self.try_index, "step_k": self.k,
             "source": self.source, "accepted": bool(self.accepted),
             "times_sampled": self.seen, "sequence": self.sequence,
             "mutated_position": None if self.mutation is None else self.mutation[0],
             "mutated_to": None if self.mutation is None else self.mutation[1],
             "balm": self.balm}
        r.update({k: v for k, v in self.scores.items()})
        r.update(extra)
        return r


# --------------------------------------------------------------------------- #
# recorder                                                                     #
# --------------------------------------------------------------------------- #
class AugmentationRecorder:
    """Records every sequence Stage 4 evaluates.

    Two hooks, both plain Python, neither of which touches the jitted graph:

      _mutate      wrapped so the (position, amino acid) of each PROPOSAL is
                   known. It also tells us that the prediction which follows is a
                   candidate rather than a baseline/replay -- design_semigreedy
                   calls predict() once before the loop starts to get the
                   starting pLDDT (af/design.py:445), and that call must not be
                   mistaken for a candidate or it competes in the argmin below.

      design/post  fires inside run(), once per predict(), i.e. once per
                   candidate. MUST be appended after the BALM callback so
                   aux["loss"] is the adjusted scalar semigreedy actually
                   minimises -- see _attach_guards in colabdesign_utils.

    Acceptance is not recorded by a hook because there is no hook to record it
    with: design_semigreedy picks argmin(loss) over the tries in Python and moves
    on. mark_accepted() replays that same argmin per iteration, which is exact
    rather than inferred.
    """

    def __init__(self, af_model, blen, balm_aux=None, score_balm=True,
                 mutation_log=None):
        self.af = af_model
        self.blen = int(blen)
        self.balm_aux = balm_aux
        self.score_balm = bool(score_balm) and balm_aux is not None
        self.mutation_log = mutation_log
        self.records = []
        self.active = False
        self.k0 = 0
        self._pending = None              # (pos, aa) from the most recent _mutate
        self._try_counter = {}
        self._by_seq = {}
        self._inner_mutate = None
        self._n_balm_fail = 0

    # -- lifecycle ---------------------------------------------------------- #
    def attach(self):
        af = self.af
        if hasattr(af, "_mutate") and self._inner_mutate is None:
            self._inner_mutate = af._mutate
            af._mutate = self._wrap_mutate(self._inner_mutate)
        af._callbacks["design"]["post"].append(self._on_predict)
        af._augment_recorder = self
        return self

    def start(self, k0=None):
        self.k0 = int(k0 if k0 is not None else getattr(self.af, "_k", 0))
        self.active = True
        return self

    def stop(self):
        self.active = False
        return self

    # -- hooks -------------------------------------------------------------- #
    def _wrap_mutate(self, inner):
        def _recording_mutate(seq, *args, **kwargs):
            out = inner(seq, *args, **kwargs)
            if not self.active:
                return out
            try:
                a, b = np.asarray(seq), np.asarray(out)
                if a.shape == b.shape:
                    a0 = a[0] if a.ndim == 2 else a
                    b0 = b[0] if b.ndim == 2 else b
                    diff = np.nonzero(a0 != b0)[0]
                    if len(diff):
                        p = int(diff[0])
                        aa = int(b0[p])
                        self._pending = (p, RESTYPES[aa] if 0 <= aa < 20 else "X")
                        if self.mutation_log is not None:
                            for q in diff:
                                self.mutation_log.add(int(getattr(self.af, "_k", 0)),
                                                      int(q), int(b0[q]))
            except Exception as e:
                print(f"[augment] could not read a proposal ({e})")
            return out
        return _recording_mutate

    def _on_predict(self, af):
        if not self.active:
            return
        try:
            seq = self._sequence_from_aux(af)
            if seq is None:
                return
            scores = self._scores_from_aux(af)
            k = int(getattr(af, "_k", 0))
            mut, self._pending = self._pending, None
            t = self._try_counter.get(k, 0)
            self._try_counter[k] = t + 1
            rec = CandidateRecord(k=k, iteration=k - self.k0 + 1, try_index=t,
                                  sequence=seq, scores=scores,
                                  balm=self._balm(seq), mutation=mut,
                                  source="mutation" if mut else "baseline")
            self.records.append(rec)
            self._by_seq.setdefault(seq, []).append(rec)
        except Exception as e:
            # never let bookkeeping kill a trajectory that is otherwise fine
            print(f"[augment] skipped a candidate ({e})")

    # -- extraction --------------------------------------------------------- #
    def _sequence_from_aux(self, af):
        """Binder-only one-letter sequence of the design just evaluated.

        Prefers "hard" (the one-hot actually fed to AF2 in the discrete stage)
        over "logits"; they agree while hard=True but "hard" is the honest one.
        """
        try:
            seq = af.aux["seq"]
            if isinstance(seq, dict):
                seq = seq.get("hard", seq.get("pseudo", seq.get("logits")))
            idx = np.asarray(seq).argmax(-1).reshape(-1)[-self.blen:]
            return "".join(RESTYPES[int(i)] if 0 <= int(i) < 20 else "X" for i in idx)
        except Exception:
            return None

    @staticmethod
    def _scores_from_aux(af):
        """The log dict run() is about to build, rebuilt here.

        Design post-callbacks fire BEFORE aux["log"] exists (af/design.py:109 vs
        :112), so the same three lines are repeated: losses is the base, "plddt"
        in losses is stored as a LOSS (1 - pLDDT) and is flipped back, and
        loss/ptm/i_ptm come off aux itself.
        """
        out = {}
        losses = af.aux.get("losses", {}) or {}
        for key, val in losses.items():
            v = _as_float(val)
            if v is not None:
                out[key] = v
        if "plddt" in out:
            out["plddt"] = 1.0 - out["plddt"]
        for key in ("loss", "ptm", "i_ptm"):
            v = _as_float(af.aux.get(key))
            if v is not None:
                out[key] = v
        for key in ("hard", "soft", "temp"):
            v = _as_float(getattr(af, "opt", {}).get(key))
            if v is not None:
                out[key] = v
        return out

    def _balm(self, seq):
        if not self.score_balm:
            return None
        try:
            return round(float(self.balm_aux.raw_score(seq)), 6)
        except Exception as e:
            self._n_balm_fail += 1
            if self._n_balm_fail <= 3:
                print(f"[augment] BALM scoring failed ({e})")
            return None

    # -- post-processing ---------------------------------------------------- #
    def mark_accepted(self):
        """Flag, per iteration, the candidate semigreedy kept: argmin of the same
        adjusted loss it compares (af/design.py:462-463). Baselines are excluded
        because they never entered that comparison."""
        by_k = {}
        for r in self.records:
            if r.source == "mutation":
                by_k.setdefault(r.k, []).append(r)
        for k, group in by_k.items():
            best = min(group, key=lambda r: r.loss)
            best.accepted = True
        return self

    def deduplicated(self):
        """One record per distinct sequence -- the best-scoring evaluation of it,
        with `seen` counting how often it came up."""
        out = []
        for seq, group in self._by_seq.items():
            best = min(group, key=lambda r: r.loss)
            best.seen = len(group)
            out.append(best)
        return out

    def summary(self):
        n_b = sum(1 for r in self.records if r.balm is not None)
        return (f"{len(self.records)} candidates, {len(self._by_seq)} unique, "
                f"{n_b} BALM-scored")


def _as_float(x):
    try:
        a = np.asarray(x)
        if a.size != 1:
            return None
        return float(a.reshape(-1)[0])
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# selection                                                                    #
# --------------------------------------------------------------------------- #
def select_candidates(records, max_negative=5):
    """Which candidates go on to MPNN. See the module docstring.

    Falls back to loss order if nothing carries a BALM score (BALM off, or
    augment_score_balm=false), so the mode still produces data in a control run.
    """
    scored = [r for r in records if r.balm is not None]
    if not scored:
        ranked = sorted(records, key=lambda r: r.loss)
        return ranked[:1]
    ranked = sorted(scored, key=lambda r: r.balm)
    if ranked[0].balm >= 0:
        return ranked[:1]
    return [r for r in ranked if r.balm < 0][:max(1, int(max_negative))]


# --------------------------------------------------------------------------- #
# filter verdict                                                               #
# --------------------------------------------------------------------------- #
def strip_filter_prefix(column):
    for p in _FILTER_PREFIXES:
        if column.startswith(p):
            return column.split("_", 1)[1]
    return column


def counted_failures(failed_columns, ignored=DEFAULT_IGNORED_FILTERS, mode="exclude"):
    """The failures that decide bad-example status.

    mode="exclude" (default): everything EXCEPT `ignored` counts.
    mode="only":              ONLY `ignored` counts.
    """
    ignored = set(ignored)
    keep = []
    for col in failed_columns or []:
        base = strip_filter_prefix(col)
        hit = base in ignored
        if (mode == "only") == hit:
            keep.append(col)
    return sorted(set(keep))


def is_bad_example(mpnn_results, ignored=DEFAULT_IGNORED_FILTERS, mode="exclude"):
    """mpnn_results: [(sequence, failed_columns_or_True), ...] -- one entry per
    MPNN redesign, where True means it passed everything.

    Returns (bad, counted_failures, all_failures). A single passing MPNN design
    makes the whole candidate a pass, per spec.
    """
    all_failed, counted = set(), set()
    for _seq, verdict in mpnn_results:
        if verdict is True:
            return False, [], []
        cols = list(verdict or [])
        all_failed.update(cols)
        counted.update(counted_failures(cols, ignored, mode))
    if not mpnn_results:
        return False, [], []
    return bool(counted), sorted(counted), sorted(all_failed)


# --------------------------------------------------------------------------- #
# CSV                                                                          #
# --------------------------------------------------------------------------- #
def append_rows(csv_path, rows):
    """Append dict rows, fixing the header on first write.

    The score columns are whatever losses were registered on the model, which is
    not knowable until the first trajectory has run -- so the header is fixed by
    the first write and later rows are reindexed onto it. Keys that appear later
    (a loss added mid-run) are not dropped; they go to `extra_json` so nothing is
    silently lost.
    """
    import pandas as pd
    if not rows:
        return 0
    with _CSV_LOCK:
        exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        if exists:
            header = list(pd.read_csv(csv_path, nrows=0).columns)
            fixed = []
            for r in rows:
                extra = {k: v for k, v in r.items() if k not in header and k != "extra_json"}
                row = {k: r.get(k) for k in header}
                if extra:
                    old = r.get("extra_json")
                    merged = json.loads(old) if isinstance(old, str) and old else {}
                    merged.update(extra)
                    if "extra_json" in header:
                        row["extra_json"] = json.dumps(merged, default=str)
                fixed.append(row)
            df = pd.DataFrame(fixed, columns=header)
        else:
            cols = []
            for r in rows:
                for k in r:
                    if k not in cols:
                        cols.append(k)
            if "extra_json" not in cols:
                cols.append("extra_json")
            df = pd.DataFrame(rows).reindex(columns=cols)
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
        df.to_csv(csv_path, mode="a", header=not exists, index=False)
        return len(df)


def write_augmentation(csv_path, records, **context):
    """One row per evaluation (or per unique sequence if the caller deduped)."""
    return append_rows(csv_path, [r.row(**context) for r in records])


def bad_example_row(record, mpnn_results, counted, all_failed, **context):
    """Columns: sequence, scores..., MPNN-seqs (list), failed_on (list)."""
    row = record.row(**context)
    seqs = [s for s, _ in mpnn_results]
    row.update({
        "n_mpnn": len(seqs),
        "mpnn_sequences": ";".join(seqs),
        "failed_on": ";".join(counted),
        "failed_on_all": ";".join(all_failed),
        # per-sequence breakdown, because the two lists above are unions and you
        # cannot tell from them which redesign failed what
        "mpnn_failures_json": json.dumps(
            {s: (list(v) if v is not True else []) for s, v in mpnn_results}, default=str),
    })
    return row


def replay_sequence(af_model, sequence, pdb_path, models=None, num_models=1,
                    num_recycles=None):
    """Re-predict one candidate and write its PDB.

    get_best=False is the whole point: save_pdb defaults to _tmp["best"], which
    is the semigreedy winner, not the candidate just predicted (af/utils.py:64-70).
    predict() saves and restores opt/params/inputs around itself, so the model is
    left exactly as it was found.
    """
    af_model.predict(seq=sequence, models=models, num_models=num_models,
                     num_recycles=num_recycles, verbose=False)
    af_model.save_pdb(pdb_path, get_best=False)
    return pdb_path


def af2_prefilter_failures(prediction_stats, filters):
    """Which AF2 prefilters a design tripped.

    predict_binder_complex computes this internally and then returns only a
    boolean, so it is re-derived here from the stats it does return -- same five
    metrics, same comparison directions, same column names.
    """
    checks = (("pLDDT", ">="), ("pTM", ">="), ("i_pTM", ">="),
              ("pAE", "<="), ("i_pAE", "<="))
    failed = []
    for model, stats in (prediction_stats or {}).items():
        for name, comparison in checks:
            col = f"{model}_{name}"
            threshold = (filters.get(col) or {}).get("threshold")
            value = (stats or {}).get(name)
            if threshold is None or value is None:
                continue
            if comparison == ">=" and value < threshold:
                failed.append(col)
            elif comparison == "<=" and value > threshold:
                failed.append(col)
    return sorted(set(failed))


def count_bad_examples(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0
    import pandas as pd
    try:
        return int(len(pd.read_csv(csv_path)))
    except Exception:
        return 0


def augment_target_reached(csv_path, target):
    n = count_bad_examples(csv_path)
    if target is None:
        return False, n
    return n >= int(target), n
