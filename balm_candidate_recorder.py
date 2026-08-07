"""
balm_candidate_recorder.py
==========================
Records every sequence the semigreedy search evaluates, ranked by loss, so that a
trajectory whose WINNER fails the post-hoc geometry checks (CA clashes, or fewer
than 3 interface residues) can fall back to the 2nd-best sequence, then the 3rd,
and so on.

WHY A RECORDER IS NEEDED
------------------------
ColabDesign's `design_semigreedy` builds a per-iteration `buff` of candidates
(af/design.py:462) and immediately discards everything except the argmin
(af/design.py:465-467). Only the single global best survives, in _tmp["best"].
There is no ranked runner-up list to fall back to, so we build one.

HOW
---
Attaches to `_callbacks["design"]["post"]`, the same plain-Python hook the BALM
term uses. It fires once per candidate evaluation, after `self.aux` is populated
and -- importantly -- AFTER the BALM callback has already adjusted
`self.aux["loss"]`, provided the BALM callback was appended first. So the ranking
this records is the same quantity semigreedy itself minimises.

Only the sequence and the scalar loss are stored (not aux), so memory is trivial.
Replaying a candidate means re-running predict() on its sequence.
"""
from __future__ import annotations

from collections import namedtuple

# k     : ColabDesign's self._k at evaluation time -> which greedy iteration
# order : 0-based call index; order 0 is design_semigreedy's seeding predict
#         (af/design.py:446), not a mutated candidate
Candidate = namedtuple("Candidate", "loss seq k order")


class CandidateRecorder:
    """Top-K ranked store of (loss, sequence) seen during a discrete stage."""

    def __init__(self, top_k=16):
        self.top_k = int(top_k)
        self._by_seq = {}      # seq bytes -> Candidate
        self.enabled = True
        self.n_seen = 0

    # -- recording -------------------------------------------------------- #
    def __call__(self, af):
        if not self.enabled:
            return
        import numpy as np
        try:
            seq = af.aux["seq"]
            if isinstance(seq, dict):
                seq = seq.get("hard", seq.get("pseudo", seq.get("logits")))
            idx = np.asarray(seq).argmax(-1)
            loss = float(af.aux["loss"])
        except Exception:
            return
        k = int(getattr(af, "_k", -1))
        key = idx.tobytes()
        prev = self._by_seq.get(key)
        # keep the BEST observed loss for a given sequence (models/recycles vary)
        if prev is None or loss < prev.loss:
            self._by_seq[key] = Candidate(loss, idx, k, self.n_seen)
        self.n_seen += 1

    # -- retrieval -------------------------------------------------------- #
    def ranked(self):
        """[Candidate, ...] ascending by loss. Rank 0 is the winner."""
        return sorted(self._by_seq.values(), key=lambda c: c.loss)

    def nth(self, n):
        """n-th best (0 = winner, 1 = runner-up, ...). None if exhausted."""
        r = self.ranked()
        return r[n] if 0 <= n < len(r) else None

    def reset(self):
        self._by_seq.clear()
        self.n_seen = 0

    def summary(self):
        r = self.ranked()
        return (f"{len(r)} unique sequences from {self.n_seen} evaluations; "
                f"loss range {r[0].loss:.2f}..{r[-1].loss:.2f}" if r else "no candidates recorded")

    def accepted_trajectory(self, k0=None):
        """The sequence ACCEPTED at each greedy iteration, in order.

        design_semigreedy evaluates `tries` mutants per iteration and keeps the
        argmin (af/design.py:465-467). Every candidate is recorded here with the
        _k it was evaluated under, so grouping by k and taking the minimum-loss
        member reconstructs exactly the accepted step -- without needing a second
        hook inside ColabDesign.

        Returns [(iteration, seq_indices, loss), ...] ascending by iteration.
        NOTE: identical sequences are deduplicated on record, keeping the best
        loss, so a sequence re-proposed in a later iteration is attributed to the
        iteration where it scored best.
        """
        best_per_k = {}
        for c in self._by_seq.values():
            if c.k < 0:
                continue
            cur = best_per_k.get(c.k)
            if cur is None or c.loss < cur.loss:
                best_per_k[c.k] = c
        out = []
        for k in sorted(best_per_k):
            c = best_per_k[k]
            it = (k - int(k0) + 1) if k0 is not None else k
            out.append((it, c.seq, c.loss))
        return out

    def iteration_of(self, cand, k0):
        """1-based greedy iteration a candidate came from. k0 = self._k captured
        immediately before design_pssm_semigreedy. Note the seeding predict
        (order 0) shares iteration 1 with the first real batch of mutants."""
        if cand is None or cand.k < 0 or k0 is None:
            return None
        return cand.k - int(k0) + 1


def attach_candidate_recorder(af, top_k=16):
    """Append a recorder to af's design post-callbacks. Returns it.

    Attach this AFTER the BALM callback so the loss it records already includes
    the BALM term; otherwise the fallback ranking would disagree with the search.
    """
    rec = CandidateRecorder(top_k=top_k)
    af._callbacks["design"]["post"].append(rec)
    return rec


def detach_candidate_recorder(af, rec):
    try:
        af._callbacks["design"]["post"].remove(rec)
    except ValueError:
        pass


def replay_candidate(af, seq, model_pdb_path, recorder=None,
                     models=None, num_models=None, num_recycles=None):
    """Re-predict `seq`, make it the model's 'best', and write it to model_pdb_path.

    clear_best() first is what makes _save_results adopt this candidate: it only
    overwrites when the new metric beats the stored one (af/design.py:269), so
    without clearing, the original winner would remain and save_pdb() -- which
    writes the BEST aux -- would keep writing the failed design.

    Recording is suspended during the replay so retries do not pollute the ranking.
    """
    was = None
    if recorder is not None:
        was, recorder.enabled = recorder.enabled, False
    try:
        af.clear_best()
        af.predict(seq=seq, models=models, num_models=num_models,
                   sample_models=False, num_recycles=num_recycles,
                   return_aux=False, verbose=False)
        af._save_results(save_best=True, verbose=False)
        af.save_pdb(model_pdb_path)
    finally:
        if recorder is not None:
            recorder.enabled = was
