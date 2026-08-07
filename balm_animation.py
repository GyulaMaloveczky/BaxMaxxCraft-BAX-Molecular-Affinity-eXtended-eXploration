"""
balm_animation.py
=================
ColabDesign's animation with a handful of additions, and NOTHING else changed:

  1. The sequence heatmap (ax2) gets amino-acid letters on the y-axis, coloured
     by side-chain property. Upstream draws imshow(seq.T) with ylabel "amino
     acids" and no tick labels, so you cannot tell which row is which residue.
  2. A per-frame STAGE label (logits / softmax / one-hot / semigreedy + the
     iteration within that stage), now drawn LARGE in its own band between the
     structure panel and the heatmap.
  3. The binder chain is outlined in the 3D panel (contour + stippled texture),
     and the video opens on a still first frame while that contour is drawn on.
  4. During Stage 4 the heatmap shows which mutations were sampled (faint blue)
     and which one was kept (full blue).

Everything else is upstream colabdesign.shared.plot.make_animation verbatim:
same plot_pseudo_3D structure panel, same pAE panel, same
ArtistAnimation(blit=True), and crucially the same ani.to_html5_video() output.

WHY to_html5_video MATTERS
    to_jshtml() embeds every frame as a base64 PNG and blew past matplotlib's
    20 MB embed limit ("Animation size has reached 21016952 bytes"). h264 keeps
    the same content to a few MB. If you want steppable frames, keep the video
    and extract them separately rather than switching the writer.

COLOUR SCHEMES (deliberately non-overlapping)
    structure / pLDDT : "afdb" -- the AlphaFold DB convention (blue = confident,
                        yellow/orange = not). Replaces upstream's gist_rainbow.
                        Chosen for instant recognisability; it is NOT perceptually
                        uniform, and yellow/orange are hard to separate under
                        protanopia/deuteranopia. plddt_cmap="viridis_r" keeps the
                        same direction and is CVD-safe if that matters more.
    sequence y-ticks  : Okabe-Ito CATEGORICAL, darkened for contrast on white.
    binder contour    : violet. Not a data colour -- it is deliberately outside
                        both the AFDB ramp (blue/yellow/orange) and the heatmap's
                        bwr_r (red/white/blue), so an outline can never be read
                        as a pLDDT value or an occupancy.
    mutation markers  : blue, per request. NOTE the heatmap itself is bwr_r with
                        vmin=-1/vmax=1, so an OCCUPIED cell is already blue. The
                        markers therefore carry a white edge and sit on top; that
                        edge, not the hue, is what separates them from the cell
                        underneath. If they still read as mushy on your designs,
                        change MUT_KEPT/MUT_TRIED below to something off-ramp
                        (e.g. the violet used for the contour) rather than
                        fiddling with alpha.

ACCESSIBILITY
    Okabe-Ito palette, and colour is redundant: every row is labelled with its
    one-letter code, so the tick text carries the information on its own. Tick
    colours are darkened variants chosen for contrast against the white axes
    background (the raw Okabe-Ito yellow is illegible there). The Stage-4
    emphasis is likewise redundant -- it flashes AND it is the only dark plaque
    AND the label still says "Stage 4", so nothing depends on noticing a colour.
"""
from __future__ import annotations

import string

import numpy as np

# ColabDesign's own helpers -- imported, not reimplemented, so the structure
# panel stays byte-for-byte the upstream rendering.
from colabdesign.shared.plot import plot_pseudo_3D, plot_ticks, pymol_cmap, jalview_color_list
from colabdesign.shared.protein import _np_kabsch

RESTYPES = "ARNDCQEGHILKMFPSTWYV"      # column order of seq[..., 20]

AA_CLASS = {}
for _aa in "AVLIM":  AA_CLASS[_aa] = "hydrophobic"
for _aa in "FWY":    AA_CLASS[_aa] = "aromatic"
for _aa in "KRH":    AA_CLASS[_aa] = "positive"
for _aa in "DE":     AA_CLASS[_aa] = "negative"
for _aa in "STNQ":   AA_CLASS[_aa] = "polar"
for _aa in "GPC":    AA_CLASS[_aa] = "special"

# Okabe-Ito, darkened where needed for contrast on a white background.
CLASS_TICK_COLOR = {
    "positive":      "#0072B2",   # #0072B2 darkened to 5.2:1 on white
    "negative":      "#BF5400",   # #D55E00 darkened to 4.7:1 on white
    "hydrophobic":   "#9C6C00",   # #E69F00 darkened to 4.6:1 on white
    "aromatic":      "#A15F83",   # #CC79A7 darkened to 4.7:1 on white
    "polar":         "#008460",   # #009E73 darkened to 4.7:1 on white
    "special":       "#7C7622",   # #F0E442 darkened to 4.7:1 on white
}
CLASS_ORDER = ["hydrophobic", "aromatic", "positive", "negative", "polar", "special"]
CLASS_LABEL = {"hydrophobic": "AVLIM", "aromatic": "FWY", "positive": "KRH",
               "negative": "DE", "polar": "STNQ", "special": "GPC"}

# Heatmap row order: PROPERTY-GROUPED rather than the raw restype order, so D/E sit
# together, K/R/H together, etc. The seq matrix rows are permuted to match, so the
# data and the labels stay in lockstep.
GROUPED_ORDER = "".join(CLASS_LABEL[c] for c in CLASS_ORDER)      # 20 letters
assert sorted(GROUPED_ORDER) == sorted(RESTYPES), "grouped order must be a permutation"
ROW_PERM = np.array([RESTYPES.index(a) for a in GROUPED_ORDER])   # display row -> restype row
ROW_INV = np.argsort(ROW_PERM)                                    # restype row -> display row
GROUP_SIZES = [len(CLASS_LABEL[c]) for c in CLASS_ORDER]
GROUP_BOUNDS = np.cumsum(GROUP_SIZES)[:-1]                        # separator positions

# --------------------------------------------------------------------------- #
# Highlight / emphasis palette (see COLOUR SCHEMES above)                      #
# --------------------------------------------------------------------------- #
HL_GLOW = "#C77DD6"        # binder contour, wide soft halo
HL_LINE = "#6A1B7A"        # binder contour, crisp rim
HL_TEXTURE = "#FFFFFF"     # stippled centre-line drawn along the binder
MUT_TRIED = "#5B9BD5"      # sampled-but-not-kept mutation (faint)
MUT_KEPT = "#0B3D91"       # the mutation semigreedy actually accepted

# Banner plaques. "Stage 4" is the only DARK one, which is most of why it reads
# as different even in a thumbnail; the flash then alternates fg/bg of that same
# plaque so the emphasis is a value inversion, not a hue change.
BANNER_BASE = dict(fc="#F4F4F6", ec="#BFBFC7", tc="#1A1A1A", lw=0.9)
BANNER_EMPH = dict(fc="#1A1A2E", ec="#FFD166", tc="#FFD166", lw=1.8)
BANNER_FLASH = dict(fc="#FFD166", ec="#1A1A2E", tc="#1A1A2E", lw=1.8)
BANNER_INTRO = dict(fc="#FFFFFF", ec=HL_LINE, tc=HL_LINE, lw=1.4)

# --------------------------------------------------------------------------- #
# pLDDT colour: the AlphaFold DB / EBI convention                              #
#   > 90  very high  #0053D6  dark blue                                        #
#   70-90 confident  #65CBF3  light blue                                       #
#   50-70 low        #FFDB13  yellow                                           #
#   < 50  very low   #FF7D45  orange                                           #
# Blue = confident, yellow/orange = not -- the opposite direction to plasma,
# and the one every reader of an AlphaFold figure already has in their head.
# plot_pseudo_3D is called with cmin=0.5, cmax=0.9, so t = (pLDDT-0.5)/0.4:
# pLDDT 0.7 lands at t=0.5, which is exactly the AFDB low/confident edge, and
# values outside the range clip to the end colours -- reproducing the
# "very low" / "very high" bands for free.
AFDB_BANDS = {"very_low": "#FF7D45", "low": "#FFDB13",
              "confident": "#65CBF3", "very_high": "#0053D6"}


# Display range. Upstream uses cmin=0.5/cmax=0.9, which CLIPS away the "very low"
# and "very high" bands entirely -- only yellow and cyan are ever reachable.
# 0.4-1.0 makes all four AFDB bands visible with their edges on the real
# thresholds: 0.5, 0.7, 0.9.
PLDDT_VMIN, PLDDT_VMAX = 0.4, 1.0


def afdb_cmap(vmin=PLDDT_VMIN, vmax=PLDDT_VMAX, discrete=True):
    """AlphaFold DB pLDDT colours, with the band edges placed at 0.5/0.7/0.9
    within the [vmin, vmax] display range.

    discrete=True  reproduces the AFDB viewer exactly: four flat bands.
    discrete=False interpolates between the same four anchor colours, so a
                   residue improving 0.72 -> 0.88 visibly changes instead of
                   sitting in one flat block. Better for watching a trajectory
                   evolve; less faithful to the standard.
    """
    from matplotlib.colors import LinearSegmentedColormap
    def t(v):
        return (v - vmin) / (vmax - vmin)
    e = 0.001
    if discrete:
        stops = [(0.0,           AFDB_BANDS["very_low"]),
                 (t(0.5) - e,    AFDB_BANDS["very_low"]),
                 (t(0.5),        AFDB_BANDS["low"]),
                 (t(0.7) - e,    AFDB_BANDS["low"]),
                 (t(0.7),        AFDB_BANDS["confident"]),
                 (t(0.9) - e,    AFDB_BANDS["confident"]),
                 (t(0.9),        AFDB_BANDS["very_high"]),
                 (1.0,           AFDB_BANDS["very_high"])]
    else:
        stops = [(0.0,      AFDB_BANDS["very_low"]),
                 (t(0.6),   AFDB_BANDS["low"]),
                 (t(0.8),   AFDB_BANDS["confident"]),
                 (1.0,      AFDB_BANDS["very_high"])]
    return LinearSegmentedColormap.from_list("afdb_plddt", stops)


# --------------------------------------------------------------------------- #
# Stage bookkeeping                                                            #
# --------------------------------------------------------------------------- #
class StageLog:
    """Records stage boundaries.

    Prefer FRAME INDEX over step index. Inferring a frame's step from af._k is
    off by one -- design_semigreedy calls _save_results and only THEN does
    self._k += 1 (af/design.py:469-474) -- and breaks entirely once
    replay_candidate appends frames via _save_results without advancing _k.
    A one-frame shift was enough to mislabel the last Stage-3 frame as
    "Stage 4 iter 1"; because that frame predates the barrier losses its logged
    loss is far lower than any real Stage-4 loss, so it wrongly won the
    min-loss search for the hold frame.

    mark(name, k, frame=len(_tmp["traj"]["seq"])) records both; frame is used
    when available."""

    def __init__(self):
        self.marks = []          # [(k, name, frame_or_None), ...]

    def mark(self, name, k, frame=None):
        self.marks.append((int(k), str(name), None if frame is None else int(frame)))

    @property
    def has_frames(self):
        return bool(self.marks) and all(m[2] is not None for m in self.marks)

    def _start_of_frame(self, f):
        start, name = None, "?"
        for _, nm, fr in self.marks:
            if fr is not None and f >= fr:
                start, name = fr, nm
            else:
                break
        return start, name

    def label_for_frame(self, f):
        start, name = self._start_of_frame(f)
        if start is None:
            return ""
        return f"{name}   (iter {f - start + 1})"

    def _start_of(self, k):
        start, name = None, "?"
        for kk, nm, _ in self.marks:
            if k >= kk:
                start, name = kk, nm
            else:
                break
        return start, name

    def stage_at(self, k):
        return self._start_of(k)[1]

    def iteration_in_stage(self, k):
        start, _ = self._start_of(k)
        return (k - start + 1) if start is not None else None

    def label_at(self, k):
        start, name = self._start_of(k)
        if start is None:
            return ""
        return f"{name}   (iter {k - start + 1})"

    def __len__(self):
        return len(self.marks)


def frame_steps(af_model, n_frames, traj_iter=None):
    """Global step _k for each trajectory frame.

    _save_results appends every traj_iter steps and caps at traj_max, dropping
    the OLDEST first (af/design.py:251-260), so frames are the TAIL of the run.
    Counting backwards from the current _k is therefore correct even after
    eviction."""
    if traj_iter is None:
        traj_iter = int(af_model._args.get("traj_iter", 1)) if hasattr(af_model, "_args") else 1
    total_k = int(getattr(af_model, "_k", n_frames * traj_iter))
    return [total_k - (n_frames - 1 - i) * traj_iter for i in range(n_frames)]


# --------------------------------------------------------------------------- #
# Sampled-mutation bookkeeping                                                 #
# --------------------------------------------------------------------------- #
class MutationLog:
    """Every mutation semigreedy PROPOSED, keyed by the global step _k.

    The trajectory only stores what was ACCEPTED -- one frame per iteration, so
    the kept mutation is just the diff between consecutive frames and needs no
    recording at all. The `tries` candidates that lost are thrown away inside
    design_semigreedy (af/design.py:455-465), so if you want to see them they
    have to be captured as they are generated.

    Keyed by _k rather than by frame index on purpose: _save_results evicts the
    OLDEST frame once traj_max is hit, which renumbers every frame, whereas _k
    only ever counts up. animate_with_stages maps frames back to _k with
    frame_steps().
    """

    def __init__(self):
        self.tried = {}                     # _k -> [(pos, aa_idx), ...]

    def add(self, k, pos, aa):
        self.tried.setdefault(int(k), []).append((int(pos), int(aa)))

    def get(self, k):
        return self.tried.get(int(k), [])

    def __len__(self):
        return len(self.tried)

    def summary(self):
        n = sum(len(v) for v in self.tried.values())
        return f"{n} sampled mutations across {len(self.tried)} steps"


def attach_mutation_recorder(af_model, mut_log=None):
    """Wrap af_model._mutate so every proposal is recorded, and hang the log off
    the model as ._mutation_log (animate_with_stages picks it up from there).

    _mutate(seq, ...) takes and returns aa INDICES of shape (N, L) -- for the
    binder protocol L is the binder, i.e. exactly the heatmap's x-axis -- so the
    proposal is recoverable as a diff without touching anything else. The
    wrapper is a plain instance attribute shadowing the bound method: no
    subclassing, no monkeypatching of the class, and `del af._mutate` undoes it.

    Returns the MutationLog, or None if this model has no _mutate (e.g. a
    gradient-only protocol) -- the animation degrades to showing just the kept
    mutation in that case.
    """
    if not hasattr(af_model, "_mutate"):
        print("[anim] af_model has no _mutate(); sampled mutations will not be recorded")
        return None
    if getattr(af_model, "_mutation_log", None) is not None and getattr(af_model, "_mutate_wrapped", False):
        return af_model._mutation_log

    log = mut_log if mut_log is not None else MutationLog()
    inner = af_model._mutate

    def _recording_mutate(seq, *args, **kwargs):
        out = inner(seq, *args, **kwargs)
        try:
            a = np.asarray(seq)
            b = np.asarray(out)
            if a.shape == b.shape:
                k = int(getattr(af_model, "_k", 0))
                # a/b are (N, L); N is 1 for every protocol BindCraft uses. Take
                # row 0 rather than any(0) so pos/aa stay paired.
                diff = np.nonzero(a[0] != b[0])[0] if a.ndim == 2 else np.nonzero(a != b)[0]
                for p in diff:
                    log.add(k, p, b[0][p] if a.ndim == 2 else b[p])
        except Exception as e:                      # recording must never break design
            print(f"[anim] mutation recorder skipped a proposal ({e})")
        return out

    af_model._mutate = _recording_mutate
    af_model._mutate_wrapped = True
    af_model._mutation_log = log
    return log


# --------------------------------------------------------------------------- #
# small drawing helpers                                                        #
# --------------------------------------------------------------------------- #
def _lw_points(ax, line_w):
    """line_w in the same units plot_pseudo_3D uses, converted to points.

    Copied from upstream (shared/plot.py:175-176) so the contour tracks the tube
    width exactly instead of drifting when the figure is resized or dpi changes.
    """
    fig = ax.get_figure()
    width_in = fig.bbox_inches.width * ax.get_position().width
    span = float(np.diff(ax.get_xlim())[0]) or 1.0
    return line_w * 72.0 * width_in / span


def _backbone_segments(xyz, chainbreak=5.0):
    """Consecutive CA-CA segments, dropping chain breaks and NaN gaps.

    Same filter as plot_pseudo_3D, but WITHOUT its O(N^2) shadow/tint pass: the
    contour is an annotation, so it wants one flat colour, not depth shading. It
    is also called once per intro frame, which the O(N^2) version would make
    needlessly slow.
    """
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 2:
        return np.zeros((0, 2, 2))
    seg = np.stack([xyz[:-1], xyz[1:]], axis=1)          # (L-1, 2, 3)
    d = np.linalg.norm(seg[:, 0] - seg[:, 1], axis=-1)
    ok = np.isfinite(d) & (d < chainbreak)
    return seg[ok][..., :2]                              # (n, 2, 2), xy only


def _contour_artists(ax, xyz, line_w, alpha=1.0, glow=HL_GLOW, line=HL_LINE):
    """Two stacked LineCollections -- wide soft halo, then a crisp rim -- drawn
    UNDER the structure (zorder < the tube's default 1) so only the outline
    shows around it."""
    from matplotlib.collections import LineCollection
    seg = _backbone_segments(xyz)
    if len(seg) == 0:
        return []
    lw = _lw_points(ax, line_w)
    out = []
    for w, colour, a, z in ((3.30, glow, 0.45 * alpha, 0.40),
                            (2.05, line, 0.95 * alpha, 0.55)):
        lc = LineCollection(seg, colors=[colour], linewidths=lw * w, alpha=a,
                            capstyle="round", joinstyle="round", zorder=z,
                            animated=True)
        ax.add_collection(lc)
        out.append(lc)
    return out


def _texture_artist(ax, xyz, line_w, alpha=0.75):
    """Stippled centre-line along the highlighted chain.

    The contour alone says "this chain"; the stipple gives it a different
    SURFACE as well, so the binder stays identifiable in a still frame even
    where the target crosses in front of it and clips the rim. Drawn on top
    (zorder 1.5) and therefore not depth-sorted against the target -- decorative,
    not geometric.
    """
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 2:
        return []
    lw = _lw_points(ax, line_w) * 0.42
    (ln,) = ax.plot(xyz[:, 0], xyz[:, 1], color=HL_TEXTURE, lw=lw,
                    ls=(0, (1.0, 1.9)), alpha=alpha, solid_capstyle="round",
                    zorder=1.5, animated=True)
    return [ln]


def _banner(ax, text, style, fontsize):
    """One plaque on the stage axes. An AXES artist, not fig.text:
    ArtistAnimation(blit=True) only redraws artists it can associate with an
    axes bbox."""
    return ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
                   fontsize=fontsize, fontweight="bold", color=style["tc"],
                   animated=True,
                   bbox=dict(boxstyle="round,pad=0.45", facecolor=style["fc"],
                             edgecolor=style["ec"], linewidth=style["lw"]))


# --------------------------------------------------------------------------- #
# make_animation -- upstream, plus the AA ticks, the stage band, the binder     #
# contour and the Stage-4 mutation overlay                                      #
# --------------------------------------------------------------------------- #
def make_animation(seq, con=None, xyz=None, plddt=None, pae=None,
                   losses=None, pos_ref=None, line_w=2.0,
                   dpi=100, interval=60, color_msa="Taylor",
                   length=None, align_xyz=True, color_by="plddt",
                   stage_labels=None, plddt_cmap="afdb", plddt_bands=True,
                   slow_stage="Stage 4", slow_factor=6, hold_frame=None,
                   hold_seconds=2.5, highlight_chain=-1, intro_seconds=2.0,
                   announce_seconds=1.2, blink_hz=3.0, sampled_mutations=None,
                   **kwargs):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.gridspec import GridSpec

    # ---------------- CHANGE 3: pLDDT colormap ---------------- #
    # Upstream leaves plot_pseudo_3D on its default "gist_rainbow": a rainbow map,
    # so neither perceptually uniform nor colourblind-safe. viridis/plasma are both.
    # Resolved to a COLORMAP OBJECT, not a string, for two reasons:
    #   * plot_pseudo_3D does matplotlib.cm.get_cmap(cmap) for strings, which was
    #     removed in matplotlib 3.9; passing an object takes the `else` branch.
    #   * its `if cmap == "gist_rainbow": c_seg *= 0.75` special case is then
    #     correctly skipped, so the full colour range is used.
    if isinstance(plddt_cmap, str):
        if plddt_cmap == "afdb":
            plddt_cmap = afdb_cmap(discrete=plddt_bands)
        else:
            try:
                plddt_cmap = matplotlib.colormaps[plddt_cmap]
            except Exception:
                plddt_cmap = plt.get_cmap(plddt_cmap)

    def nankabsch(a, b, **kw):
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[ok], b[ok]
        return _np_kabsch(a, b, **kw)

    if xyz is not None:
        if pos_ref is None:
            pos_ref = xyz[-1]
        if length is None:
            L, Ls = len(pos_ref), None
        elif isinstance(length, list):
            L, Ls = length[0], length
        else:
            L, Ls = length, None

        if align_xyz:
            pos_ref_trim = pos_ref[:L]
            pos_ref_trim_mu = np.nanmean(pos_ref_trim, 0)
            pos_ref_trim = pos_ref_trim - pos_ref_trim_mu
            new_pos = []
            for x in xyz:
                x_mu = np.nanmean(x[:L], 0)
                aln = nankabsch(x[:L] - x_mu, pos_ref_trim, use_jax=False)
                new_pos.append((x - x_mu) @ aln)
            pos = np.array(new_pos)
            pos_mean = np.concatenate(pos, 0)
            m = np.nanmean(pos_mean, 0)
            rot_mtx = nankabsch(pos_mean - m, pos_mean - m, return_v=True, use_jax=False)
            pos = (pos - m) @ rot_mtx
            pos_ref_full = ((pos_ref - pos_ref_trim_mu) - m) @ rot_mtx
        else:
            pos_mean = np.concatenate(xyz, 0)
            m = np.nanmean(pos_mean, 0)
            aln = nankabsch(pos_mean - m, pos_mean - m, return_v=True, use_jax=False)
            pos = [(x - m) @ aln for x in xyz]
            pos_ref_full = (pos_ref - m) @ aln

    if pae is not None and len(pae) == 0:
        pae = None
    fig = plt.figure()
    # CHANGE 4: 12 rows instead of upstream's 4. The extra resolution buys a
    # DEDICATED ROW for the stage banner, sitting between the structure panel and
    # the heatmap rather than floating inside the structure panel where it used
    # to overlap the protein. Row budget: 6 structure / 1 banner / 5 heatmap, so
    # the heatmap still gets more height than upstream's 1-in-4 -- room for
    # readable amino-acid labels.
    gs = GridSpec(12, 3, figure=fig)
    if pae is not None:
        ax1, ax2, ax3 = fig.add_subplot(gs[:6, :2]), fig.add_subplot(gs[7:, :]), fig.add_subplot(gs[:6, 2:])
    else:
        ax1, ax2 = fig.add_subplot(gs[:6, :]), fig.add_subplot(gs[7:, :])
    axS = fig.add_subplot(gs[6, :])                 # stage banner band
    axS.set_axis_off()
    axS.set_xlim(0, 1); axS.set_ylim(0, 1)

    fig.subplots_adjust(top=0.95, bottom=0.09, right=0.95, left=0.06, hspace=0.35, wspace=0)
    # figheight grew 6.8 -> 7.4 to pay for the banner row: without it the band
    # would come out of the structure panel's height.
    fig.set_figwidth(8); fig.set_figheight(7.4); fig.set_dpi(dpi)
    ax2.set_xlabel("positions")
    single_seq = seq[0].shape[0] == 1
    if not single_seq:
        ax2.set_ylabel("sequences")
        ax2.set_yticks([])
    else:
        # ---------------- CHANGE 1: label the amino-acid axis ---------------- #
        # imshow(seq[k][0].T) puts the 20 restypes on y in RESTYPES order.
        ax2.set_ylabel("amino acids", fontsize=9)
        ax2.set_yticks(range(20))
        ax2.set_yticklabels(list(GROUPED_ORDER), fontsize=7.5, fontweight="bold")
        for lab in ax2.get_yticklabels():
            cls = AA_CLASS.get(lab.get_text())
            if cls:
                lab.set_color(CLASS_TICK_COLOR[cls])
        ax2.tick_params(axis="y", length=2, pad=1.5)
        # thin rules between property blocks
        for b in GROUP_BOUNDS:
            ax2.axhline(b - 0.5, color="#666", lw=0.6, alpha=0.55)
        # compact property key, drawn once (colour is redundant with the letters)
        key = "   ".join(f"{CLASS_LABEL[c]}" for c in CLASS_ORDER)
        x0 = 0.0
        for c in CLASS_ORDER:
            t = ax2.text(x0, -0.20, CLASS_LABEL[c], transform=ax2.transAxes,
                         fontsize=7, fontweight="bold", color=CLASS_TICK_COLOR[c],
                         va="top", ha="left")
            x0 += 0.045 + 0.016 * len(CLASS_LABEL[c])

    if xyz is None:
        ax1.set_title("predicted contact map")
    else:
        if plddt is None:
            ax1.set_title("N→C")
        else:
            ax1.set_title("pLDDT", fontsize=10)
    if pae is not None:
        ax3.set_title("pAE"); ax3.set_xticks([]); ax3.set_yticks([])

    if xyz is not None and plddt is not None and color_by == "plddt":
        # CHANGE 5: colorbar for pLDDT. An INSET axes inside ax1 rather than
        # fig.colorbar(ax=ax1), which would resize ax1 and break the hand-tuned
        # subplots_adjust the upstream layout depends on.
        # Bar sits well inside ax1 with its TICKS ON TOP: with ticks underneath at
        # y=0.045 the numbers fell outside the axes and were clipped in half.
        cax = ax1.inset_axes([0.045, 0.075, 0.32, 0.024])
        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(PLDDT_VMIN, PLDDT_VMAX),
                                          cmap=plddt_cmap)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([0.4, 0.5, 0.7, 0.9, 1.0])
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.outline.set_linewidth(0.5)
        for t in cb.ax.get_xticklabels():
            t.set_clip_on(False)
        # caption below the bar, still inside ax1
        ax1.text(0.045, 0.062, "pLDDT", transform=ax1.transAxes, fontsize=7,
                 va="top", ha="left", color="#222")

    if xyz is not None:
        main_pos = pos_ref_full[np.isfinite(pos_ref_full).all(1)]
        pred_pos = [np.isfinite(x).all(1) for x in pos]
        x_min, y_min, z_min = np.minimum(np.mean([x.min(0) for x in pred_pos], 0), main_pos.min(0)) - 5
        x_max, y_max, z_max = np.maximum(np.mean([x.max(0) for x in pred_pos], 0), main_pos.max(0)) + 5
        x_pad = ((y_max - y_min) * 2 - (x_max - x_min)) / 2
        y_pad = ((x_max - x_min) / 2 - (y_max - y_min)) / 2
        if x_pad > 0:
            x_min -= x_pad; x_max += x_pad
        else:
            y_min -= y_pad; y_max += y_pad
        ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max)
    ax1.set_xticks([]); ax1.set_yticks([])

    # ------------------------------------------------------------------ #
    # CHANGE 8: which residues are the binder?                            #
    # Ls is [target_len, binder_len] for the binder protocol, so the binder
    # is the LAST chain; highlight_chain indexes Ls (-1 = last, None = off).
    # ------------------------------------------------------------------ #
    hl = None
    hl_letter = ""
    if xyz is not None and highlight_chain is not None and isinstance(length, list) and len(length) > 1:
        bounds = np.cumsum([0] + [int(v) for v in length])
        ci = int(highlight_chain) % len(length)
        hl = (int(bounds[ci]), int(bounds[ci + 1]))
        hl_letter = string.ascii_uppercase[ci] if ci < 26 else "?"

    # marker geometry for the heatmap overlay: one cell is 1 unit in both axes,
    # so size the dots off the cell, not off a fixed number of points.
    if single_seq:
        w_in = fig.bbox_inches.width * ax2.get_position().width
        h_in = fig.bbox_inches.height * ax2.get_position().height
        n_pos = seq[0].shape[1]
        cell = min(w_in * 72.0 / max(n_pos, 1), h_in * 72.0 / 20.0)
        d_tried = max(3.2, cell * 0.90)
        d_kept = max(4.5, cell * 1.25)

    def _mut_marks(muts, kind):
        """Heatmap dots for [(pos, aa_idx), ...]. Rows are permuted for display,
        so the restype index has to go through ROW_INV to land on the right row.

        The kept marker is a BULLSEYE -- white ring outside, saturated blue
        inside -- because the cell underneath it is itself blue (bwr_r puts an
        occupied residue at the blue end), so a plain blue dot would sit on a
        blue square and disappear. The ring is what does the separating."""
        if not single_seq or not muts:
            return []
        xs = [int(p) for p, _ in muts]
        ys = [int(ROW_INV[int(a)]) for _, a in muts]
        if kind == "tried":
            return [ax2.scatter(xs, ys, s=d_tried ** 2, marker="o", c=[MUT_TRIED],
                                alpha=0.55, linewidths=0.35, edgecolors="white",
                                animated=True, zorder=4, clip_on=True)]
        art = []
        if kind == "kept":
            for x in xs:
                (ln,) = ax2.plot([x, x], [-0.5, 19.5], color=MUT_KEPT, lw=0.8,
                                 alpha=0.45, animated=True, zorder=5)
                art.append(ln)
            art.append(ax2.scatter(xs, ys, s=(d_kept * 1.45) ** 2, marker="o",
                                   facecolors="none", edgecolors="white", linewidths=1.3,
                                   animated=True, zorder=6, clip_on=True))
        art.append(ax2.scatter(xs, ys, s=d_kept ** 2, marker="o", c=[MUT_KEPT],
                               alpha=1.0 if kind == "kept" else 0.40,
                               linewidths=0.8 if kind == "kept" else 0.0,
                               edgecolors="white", animated=True, zorder=7,
                               clip_on=True))
        return art

    # accepted mutation per frame = diff of consecutive one-hot argmaxes. The
    # trajectory already contains this; nothing needs to be recorded for it.
    kept_by_frame = {}
    if single_seq:
        idx = np.asarray([np.asarray(s[0]).argmax(-1) for s in seq])
        for k in range(1, len(idx)):
            ch = np.nonzero(idx[k] != idx[k - 1])[0]
            if len(ch):
                kept_by_frame[k] = [(int(p), int(idx[k][p])) for p in ch]

    def _is_slow(k):
        return (stage_labels is not None and k < len(stage_labels)
                and slow_stage and stage_labels[k].startswith(slow_stage))

    base, halo, banner, overlay = [], [], [], []
    for k in range(len(seq)):
        arts = []
        if xyz is not None:
            flags = dict(ax=ax1, line_w=line_w, zmin=z_min, zmax=z_max)
            if color_by == "plddt" and plddt is not None:
                arts.append(plot_pseudo_3D(pos[k], c=plddt[k], Ls=Ls,
                                           cmin=PLDDT_VMIN, cmax=PLDDT_VMAX,
                                           cmap=plddt_cmap, **flags))
            elif color_by == "chain":
                c = np.concatenate([[n] * L for n, L in enumerate(length)])
                arts.append(plot_pseudo_3D(pos[k], c=c, Ls=Ls, cmap=pymol_cmap, cmin=0, cmax=39, **flags))
            else:
                L = pos[k].shape[0]
                arts.append(plot_pseudo_3D(pos[k], c=np.arange(L)[::-1], Ls=Ls, cmin=0, cmax=L, **flags))
        else:
            L = con[k].shape[0]
            arts.append(ax1.imshow(con[k], animated=True, cmap="Greys", vmin=0, vmax=1, extent=(0, L, L, 0)))

        if single_seq:
            # rows permuted into property groups; aspect="auto" so the heatmap
            # fills the taller panel instead of keeping square pixels
            arts.append(ax2.imshow(seq[k][0].T[ROW_PERM], animated=True, cmap="bwr_r",
                                   vmin=-1, vmax=1, aspect="auto"))
        else:
            import matplotlib as mpl
            cmap = mpl.colors.ListedColormap(jalview_color_list[color_msa])
            vmax = len(jalview_color_list[color_msa]) - 1
            arts.append(ax2.imshow(seq[k].argmax(-1), animated=True, cmap=cmap, vmin=0,
                                   vmax=vmax, interpolation="none", aspect="auto"))

        if pae is not None:
            L = pae[k].shape[0]
            arts.append(ax3.imshow(pae[k], animated=True, cmap="bwr", vmin=0, vmax=30, extent=(0, L, L, 0)))
        base.append(arts)

        # CHANGE 8 (cont.): contour + stipple, rebuilt per frame because the
        # binder moves. Kept in a SEPARATE list so the intro can show frame 0
        # without it and draw it on progressively.
        if hl is not None:
            sub = pos[k][hl[0]:hl[1]]
            halo.append(_contour_artists(ax1, sub, line_w) + _texture_artist(ax1, sub, line_w))
        else:
            halo.append([])

        # ---------------- CHANGE 2: per-frame stage label ---------------- #
        # Now on its own axes between the two panels, and much larger.
        if stage_labels is not None and k < len(stage_labels):
            emph = bool(slow_stage) and stage_labels[k].startswith(slow_stage)
            banner.append([_banner(axS, stage_labels[k],
                                   BANNER_EMPH if emph else BANNER_BASE,
                                   16 if emph else 14)])
        else:
            banner.append([])

        # ---------------- CHANGE 9: Stage-4 mutation overlay ---------------- #
        if _is_slow(k):
            tried = list(sampled_mutations.get(k, [])) if sampled_mutations else []
            kept = kept_by_frame.get(k, [])
            # do not draw a sampled dot underneath the kept one
            tried = [m for m in tried if m not in kept]
            overlay.append({"tried": _mut_marks(tried, "tried"),
                            "dim": _mut_marks(kept, "dim"),
                            "kept": _mut_marks(kept, "kept")})
        else:
            overlay.append(None)

    if length is not None:
        Ls = length if isinstance(length, list) else [length, None]
        if con is not None:
            plot_ticks(ax1, Ls, con[0].shape[0])
        if pae is not None:
            plot_ticks(ax3, Ls, pae[0].shape[0])

    def _full(k):
        """Everything frame k ends up showing once its reveal has finished."""
        o = overlay[k]
        extra = (o["tried"] + o["kept"]) if o else []
        return base[k] + halo[k] + banner[k] + extra

    ims = []

    # ------------------------------------------------------------------ #
    # CHANGE 10: opening. Hold on the still first frame, draw the contour
    # around the binder one segment at a time, settle, then run. The contour
    # artists for frame 0 are the same objects reused later, so it does not
    # "re-appear" when the trajectory starts -- it simply stays.
    # ------------------------------------------------------------------ #
    if hl is not None and intro_seconds and intro_seconds > 0:
        n_intro = max(3, int(round(intro_seconds * 1000.0 / max(interval, 1))))
        n_still = max(1, int(round(n_intro * 0.22)))
        n_settle = max(1, int(round(n_intro * 0.18)))
        n_draw = max(1, n_intro - n_still - n_settle)
        intro_banner = [_banner(axS, f"binder — chain {hl_letter}", BANNER_INTRO, 14)]
        sub0 = pos[0][hl[0]:hl[1]]
        ims.extend([base[0] + intro_banner] * n_still)
        for i in range(1, n_draw + 1):
            m = max(2, int(round(len(sub0) * i / n_draw)))
            grow = _contour_artists(ax1, sub0[:m], line_w)
            ims.append(base[0] + grow + intro_banner)
        # halo[0] is contour + stipple, i.e. exactly what every later frame shows
        ims.extend([base[0] + halo[0] + intro_banner] * n_settle)

    # ------------------------------------------------------------------ #
    # CHANGE 6: slow down one stage. ArtistAnimation has a single `interval` for
    # the whole animation, so per-stage speed is achieved by REPEATING that
    # stage's frames -- 6 sub-frames at 60 ms each == 360 ms on screen. The
    # repeats are no longer identical: they are the reveal (sampled -> kept).
    # CHANGE 11: announce Stage 4. The trajectory freezes on the frame it is
    # about to leave while the banner flashes, so the switch to the discrete
    # search is impossible to miss.
    # ------------------------------------------------------------------ #
    first_slow = next((i for i in range(len(base)) if _is_slow(i)), None)
    n_announce = max(0, int(round(announce_seconds * 1000.0 / max(interval, 1)))) if first_slow is not None else 0
    announce = []
    if n_announce:
        half = max(1, int(round(500.0 / max(blink_hz, 0.1) / max(interval, 1))))
        name = stage_labels[first_slow].split("(")[0].strip()
        flash = [[_banner(axS, name, BANNER_EMPH, 17)],
                 [_banner(axS, name, BANNER_FLASH, 17)]]
        f = max(0, first_slow - 1)                       # the state Stage 4 starts from
        announce = [base[f] + halo[f] + flash[(j // half) % 2] for j in range(n_announce)]

    for k in range(len(base)):
        if k == first_slow and announce:
            # freeze + flash AT the boundary, not before the trajectory starts
            ims.extend(announce)
        if not _is_slow(k):
            ims.append(base[k] + halo[k] + banner[k])
            continue
        o = overlay[k]
        n_sub = max(1, int(slow_factor or 1))
        # first ~third: what was sampled. one transition frame where the winner
        # fades up. the rest: the winner at full strength.
        n_tried = 0 if n_sub == 1 else max(1, n_sub // 3)
        n_dim = 1 if n_sub >= 3 else 0
        for r in range(n_sub):
            if r < n_tried:
                phase = "tried"
            elif r < n_tried + n_dim:
                phase = "dim"
            else:
                phase = "kept"
            extra = []
            if o:
                extra += o["tried"]
                if phase == "dim":
                    extra += o["dim"]
                elif phase == "kept":
                    extra += o["kept"]
            ims.append(base[k] + halo[k] + banner[k] + extra)

    # CHANGE 7: hold on the chosen frame at the end, so the video visibly stops on
    # the design that was actually kept rather than on whatever came last.
    if hold_frame is not None and 0 <= hold_frame < len(base):
        n_hold = max(1, int(round(hold_seconds * 1000.0 / max(interval, 1))))
        ims.extend([_full(hold_frame)] * n_hold)

    ani = animation.ArtistAnimation(fig, ims, blit=True, interval=interval)
    plt.close()
    return ani.to_html5_video()


# --------------------------------------------------------------------------- #
# Drop-in for af_model.animate(dpi=...)                                        #
# --------------------------------------------------------------------------- #
def animate_with_stages(af_model, stage_log=None, s=0, e=None, dpi=100,
                        get_best=True, traj=None, aux=None, color_by="plddt",
                        plddt_cmap="afdb", plddt_bands=False,
                        slow_stage="Stage 4", slow_factor=6,
                        hold_seconds=2.5, highlight_chain=-1,
                        intro_seconds=2.0, announce_seconds=1.2,
                        mutation_log=None):
    """Mirrors colabdesign.af.utils._af_utils.animate() exactly, then adds the
    stage labels. Same arguments, same return type (html5 video string).

    mutation_log: a MutationLog from attach_mutation_recorder(). Defaults to
    af_model._mutation_log if the recorder was attached, and to None otherwise --
    in which case Stage 4 still highlights the mutation it KEPT, just not the
    ones it sampled and rejected (that information does not survive
    design_semigreedy)."""
    if aux is None:
        aux = af_model._tmp["best"]["aux"] if (get_best and "aux" in af_model._tmp["best"]) else af_model.aux
    aux = aux["all"]
    if af_model.protocol in ["fixbb", "binder"]:
        pos_ref = af_model._inputs["batch"]["all_atom_positions"][:, 1].copy()
        pos_ref[(pos_ref == 0).any(-1)] = np.nan
    else:
        pos_ref = aux["atom_positions"][0, :, 1, :]

    if traj is None:
        traj = af_model._tmp["traj"]
    sub_traj = {k: v[s:e] for k, v in traj.items()}

    labels = None
    if stage_log is not None and len(stage_log):
        n_all = len(traj["seq"])
        if stage_log.has_frames:
            # exact: boundaries were recorded as frame indices
            labels = [stage_log.label_for_frame(i) for i in range(n_all)][s:e]
        else:
            # legacy fallback: infer from _k. -1 because _save_results runs BEFORE
            # self._k += 1, so the final frame belongs to step _k-1, not _k.
            ks = [k - 1 for k in frame_steps(af_model, n_all)]
            labels = [stage_log.label_at(k) for k in ks][s:e]

    # Sampled mutations are keyed by _k (frame indices shift when traj_max evicts
    # from the front), so map them back through the same -1 step correction.
    if mutation_log is None:
        mutation_log = getattr(af_model, "_mutation_log", None)
    sampled = None
    if mutation_log is not None and len(mutation_log):
        try:
            n_all = len(traj["seq"])
            ks = [k - 1 for k in frame_steps(af_model, n_all)]
            sampled = {i: mutation_log.get(ks[i]) for i in range(n_all)}
            sampled = {i - s: v for i, v in sampled.items()
                       if v and i >= s and (e is None or i < e)}
        except Exception as _e:
            print(f"[anim] could not align sampled mutations ({_e}); showing kept only")
            sampled = None

    # Which frame is the design that actually gets kept? best_metric is "loss" and
    # clear_best() runs at the start of Stage 4, so _tmp["best"] is the Stage-4
    # argmin. _tmp["log"] has one entry per _save_results call; take its tail so it
    # lines up with the (possibly evicted, possibly subsampled) frame list.
    hold_frame = None
    if labels is not None:
        try:
            losses = np.asarray(af_model.get_loss("loss"), dtype=float)
            n = len(sub_traj["seq"])
            if len(losses) >= n:
                lf = losses[-n:]
                cand = [i for i, lb in enumerate(labels) if lb.startswith(slow_stage)]
                if cand:
                    hold_frame = int(min(cand, key=lambda i: lf[i]))
        except Exception as e:
            print(f"[anim] could not locate chosen frame ({e}); no hold applied")

    align_xyz = af_model.protocol == "hallucination"
    return make_animation(**sub_traj, pos_ref=pos_ref, length=af_model._lengths,
                          color_by=color_by, align_xyz=align_xyz, dpi=dpi,
                          stage_labels=labels, plddt_cmap=plddt_cmap, plddt_bands=plddt_bands,
                          slow_stage=slow_stage, slow_factor=slow_factor,
                          hold_frame=hold_frame, hold_seconds=hold_seconds,
                          highlight_chain=highlight_chain, intro_seconds=intro_seconds,
                          announce_seconds=announce_seconds, sampled_mutations=sampled)