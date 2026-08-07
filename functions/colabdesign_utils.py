####################################
############## ColabDesign functions
####################################
### Import dependencies
import os, re, shutil, math, pickle, time
# --- persistent XLA compilation cache: MUST be configured before JAX compiles ---
# ColabDesign rebuilds af_model every trajectory, so the in-memory jit cache always
# misses (it is keyed on the Python function object). The persistent cache is keyed
# on the HLO hash, so an identical graph from a fresh model object hits it and AF2
# compiles once per distinct graph for the whole run instead of once per trajectory.
# NOTE: exporting JAX_COMPILATION_CACHE_DIR in the SLURM script is the robust route,
# because it is honoured at JAX init regardless of import order. This import is a
# fallback for when colabdesign_utils happens to be the first thing to touch JAX.
try:
    import balm_jax_cache
    balm_jax_cache.configure()
    _JAX_CACHE = True
except Exception as _e:
    _JAX_CACHE = False
    print(f"[jax-cache] unavailable ({_e}); AF2 will recompile every trajectory.")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from colabdesign.shared.utils import copy_dict
from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages
from .pyrosetta_utils import pr_relax, align_pdbs
from .generic_utils import update_failures
from balm_aux_loss import BalmAuxLoss
# BALM is attached as a ColabDesign *design* post-callback (plain Python, fires
# after the jitted call returns), NOT as a traced model loss. The old traced route
# via jax.pure_callback cost ~0.55s of device stall per invocation and made
# jit(_model) ~6x slower to compile; measured Stage 4 was ~292s vs ~69s without.
from balm_design_callback import add_balm_design_callback, remove_balm_design_callback
from balm_hydrophobic_guidance import add_interface_hydrophobic_barrier_loss, add_hydrophobic_fraction_barrier_loss
# structure barrier (pLDDT/i_ptm floor) + optional ESM naturalness guard that keep
# the BALM-steered search on the foldable, in-distribution manifold.
from balm_structure_guidance import add_structure_barrier_loss, add_esm_naturalness_loss
# atom-atom (5A) interface contact wall -- measures the SAME thing hotspot_residues
# does, unlike i_con which is CB-CB at 10A and is satisfied by a peptide 7-8A away.
from balm_contact_guidance import add_interface_contact_barrier_loss
# ranked runner-up store, so a winner that fails geometry can fall back to 2nd/3rd best
from balm_candidate_recorder import attach_candidate_recorder, replay_candidate
# richer trajectory animation: stage banner + per-residue AA letters coloured by
# side-chain property (colourblind-safe, letter always drawn so colour is redundant)
from balm_animation import animate_with_stages, StageLog, MutationLog, attach_mutation_recorder
# "Augment sampling" mode: keep every sequence Stage 4 evaluates instead of only
# the winner. Writes Augmentation.csv / Bad_examples.csv; see augment_sampling.py.
from .augment_sampling import (augment_settings, augment_paths, AugmentationRecorder,
                               select_candidates, write_augmentation)

# Profiling: counts + wall time for the BALM host callback. Optional -- if
# balm_profile.py is absent the run proceeds unprofiled rather than crashing.
try:
    from balm_profile import wrap_method, report as _balm_report_stats, reset as _balm_reset_stats
    _BALM_PROFILE = True
except Exception:
    _BALM_PROFILE = False
    def wrap_method(obj, attr, name=None): return getattr(obj, attr)
    def _balm_report_stats(**kwargs): pass
    def _balm_reset_stats(): pass

# BALM config lives at the repo root (one level up from functions/); resolve it
# absolutely so it works regardless of the working directory (e.g. under SLURM).
_BALM_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "balm_config.json")
# Minimum pLDDT the (Stage-3) starting design must have BEFORE Stage-4 semigreedy is
# allowed to start. Keep this equal to the structure-barrier plddt_floor so BALM only
# ever steers designs that already clear the wall (instead of fighting up from below it).
STAGE4_START_PLDDT = 0.6
# How many runner-up sequences to try when the semigreedy winner fails the CA-clash
# or interface-contact check. Each retry costs one AF2 predict (~1.4s), so 5 is cheap
# against a ~180s trajectory that would otherwise be discarded entirely.
MAX_RUNNERUP_RETRIES = 5
# Canonical residue order (matches "".join(residue_constants.restypes))
_RESTYPES = "ARNDCQEGHILKMFPSTWYV"


def get_balm_scorer():
    """The BALM singleton, or None. Lets bindcraft.py rescore MPNN sequences with
    the SAME cached model and target embedding used during design:
        s = get_balm_scorer(); cos = s.raw_score(binder_sequence)
    Results are cached, so re-scoring a sequence already seen is free."""
    return getattr(binder_hallucination, "_balm_aux", None)


def _binder_seq_from_aux(af, blen):
    """Binder sequence in af.aux -- what design_semigreedy starts from
    (it reads aux["seq"]["logits"].argmax(-1), af/design.py:435). Captured BEFORE
    clear_best(), which wipes _tmp["best"] but leaves aux untouched."""
    try:
        seq = af.aux["seq"]
        if isinstance(seq, dict):
            seq = seq.get("logits", seq.get("hard", seq.get("pseudo")))
        idx = np.asarray(seq).argmax(-1).reshape(-1)[-blen:]
        return "".join(_RESTYPES[int(i)] if 0 <= int(i) < 20 else "X" for i in idx)
    except Exception:
        return None


def _seq_from_indices(idx, blen):
    idx = np.asarray(idx).reshape(-1)[-blen:]
    return "".join(_RESTYPES[int(i)] if 0 <= int(i) < 20 else "X" for i in idx)


def _binder_seq_from_best(af, blen):
    """Binder sequence of the design held in _tmp['best'] -- i.e. exactly what
    save_pdb() wrote, including a runner-up promoted by the fallback."""
    try:
        seq = af._tmp["best"]["aux"]["seq"]
        if isinstance(seq, dict):
            seq = seq.get("hard", seq.get("pseudo", seq.get("logits")))
        idx = np.asarray(seq).argmax(-1).reshape(-1)[-blen:]
        return "".join(_RESTYPES[int(i)] if 0 <= int(i) < 20 else "X" for i in idx)
    except Exception:
        return None


def _balm_profile_report(tag, expected=None):
    """Print the BALM host-callback profile for the trajectory just finished, then
    reset the counters so the next trajectory is measured independently.

    expected: predicted callback count. A recorded count far above this means the
    loss is firing outside the discrete stage (each such call is a full 650M ESM
    pass plus a pure_callback device stall, for zero gradient benefit)."""
    if not _BALM_PROFILE:
        return
    print(f"[balm-prof] --- {tag} ---")
    _balm_report_stats(expected=expected)
    _balm_reset_stats()


def _balm_env_banner(state={}):
    """Print the XLA allocator settings once per process. An empty dict here means
    the vars never reached Python, and JAX is using its defaults (preallocate=true,
    ~75% of VRAM) regardless of what the SLURM script says."""
    if state.get("done"):
        return
    state["done"] = True
    xla = {k: v for k, v in os.environ.items() if "XLA_PYTHON" in k}
    print(f"[balm] XLA env: {xla if xla else 'NONE SET (JAX defaults in effect)'}")
    if _JAX_CACHE:
        print(f"[jax-cache] status: {balm_jax_cache.status()}")


def _compile_report(tag):
    """Report XLA compiles for the trajectory just finished, then reset.

    With a fixed binder length and a working cache: trajectory 1 shows a handful,
    every trajectory after shows 0. A steady nonzero count means the cache is not
    hitting -- check that the dir is writable and that no shape is varying."""
    if _JAX_CACHE:
        balm_jax_cache.report_compiles(tag)


# hallucinate a binder
def binder_hallucination(design_name, starting_pdb, chain, target_hotspot_residues, length, seed, helicity_value, design_models, advanced_settings, design_paths, failure_csv):
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")
    _t_traj = time.perf_counter()          # wall-clock breakdown (see [timing] lines)
    # clear GPU memory for new trajectory
    clear_mem()
    # initialise binder hallucination model
    af_model = mk_afdesign_model(protocol="binder", debug=False, data_dir=advanced_settings["af_params_dir"],
                                use_multimer=advanced_settings["use_multimer_design"], num_recycles=advanced_settings["num_recycles_design"],
                                best_metric='loss')
    # build the BALM affinity scorer once (loads the 650M model lazily on first score),
    # and cache it on the function so it is reused across trajectories.
    balm_aux = getattr(binder_hallucination, "_balm_aux", None)
    if balm_aux is None:
        balm_aux = BalmAuxLoss.from_json(_BALM_CONFIG_PATH)
        balm_aux.run_sanity_checks(load_model=False)   # cheap config/gating pre-flight
        binder_hallucination._balm_aux = balm_aux
        # multi-target only: set balm_aux.cfg.target_sequence = <target chain seq> here.
        # --- profiling instrumentation (once, on the singleton) ---
        # host_penalty is already decorated inside balm_colabdesign_bridge; these two
        # add the layers beneath it so the report separates cache hits from real ESM
        # passes. Monkeypatching works because Python resolves obj.attr at call time.
        wrap_method(balm_aux, "raw_score")
        wrap_method(balm_aux.predictor, "score")
        if _JAX_CACHE:
            balm_jax_cache.install_compile_counter()
        _balm_env_banner()
    # sanity check for hotspots
    if target_hotspot_residues == "":
        target_hotspot_residues = None
    af_model.prep_inputs(pdb_filename=starting_pdb, chain=chain, binder_len=length, hotspot=target_hotspot_residues, seed=seed, rm_aa=advanced_settings["omit_AAs"],
                        rm_target_seq=advanced_settings["rm_template_seq_design"], rm_target_sc=advanced_settings["rm_template_sc_design"])
    ### Update weights based on specified settings
    af_model.opt["weights"].update({"pae":advanced_settings["weights_pae_intra"],
                                    "plddt":advanced_settings["weights_plddt"],
                                    "i_pae":advanced_settings["weights_pae_inter"],
                                    "con":advanced_settings["weights_con_intra"],
                                    "i_con":advanced_settings["weights_con_inter"],
                                    })
    # redefine intramolecular contacts (con) and intermolecular contacts (i_con) definitions
    af_model.opt["con"].update({"num":advanced_settings["intra_contact_number"],"cutoff":advanced_settings["intra_contact_distance"],"binary":False,"seqsep":9})
    af_model.opt["i_con"].update({"num":advanced_settings["inter_contact_number"],"cutoff":advanced_settings["inter_contact_distance"],"binary":False})
    ### additional loss functions
    if advanced_settings["use_rg_loss"]:
        # radius of gyration loss
        add_rg_loss(af_model, advanced_settings["weights_rg"])
    if advanced_settings["use_i_ptm_loss"]:
        # interface pTM loss
        add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])
    if advanced_settings["use_termini_distance_loss"]:
        # termini distance loss
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])
    # add the helicity loss
    add_helix_loss(af_model, helicity_value)
    # calculate the number of mutations to do based on the length of the protein
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))
    # is the BALM affinity term active? (discrete stages only)
    balm_on = balm_aux.cfg.enabled and balm_aux.cfg.weight != 0.0
    # Are the structure/hydrophobicity WALLS active? Deliberately INDEPENDENT of
    # balm_on so that "BALM off" is a clean control: identical loss surface, identical
    # stage-4 gating and clear_best, identical accept/reject policy -- the ONLY
    # difference being the BALM term itself. Previously the walls were attached inside
    # `if balm_on:`, so a weight=0 run silently dropped the barriers too and was not a
    # valid comparison. Override with advanced_settings["use_balm_guards"] = false.
    guards_on = bool(advanced_settings.get("use_balm_guards", True))
    # Atom-atom (5A) interface contact wall. OFF by default: measured at ~0.34s per
    # candidate (Stage 4 went 0.39 -> 0.73 s/candidate when it was added), and on the
    # three trajectories that reached Stage 4 with it enabled it did NOT prevent the
    # post-relax "no interface residues" rejection -- it constrains the AF2 structure
    # while that check runs on the PyRosetta-relaxed one. Enable with
    # advanced_settings["use_contact_barrier"] = true.
    contact_barrier_on = bool(advanced_settings.get("use_contact_barrier", False))
    # BALM-only acceptance policy (applies when balm_on): keep a trajectory ONLY if its
    # final design came out of the BALM-active discrete semigreedy search. `stage4_ran`
    # is flipped True the moment that search actually executes; anything still False at
    # the end is force-rejected below.
    stage4_ran = False
    # Models evaluated per semigreedy step. >1 averages out the single-model pLDDT noise
    # that random-walks the search into low-confidence space (slower, but steadier walk &
    # a more robust save_best). Override in settings via advanced_settings["stage4_num_models"].
    stage4_num_models = int(advanced_settings.get("stage4_num_models", 1))
    # Expected BALM host-callback count for a discrete stage: the loss runs once per
    # model per candidate, so num_models multiplies the obvious iters*tries figure.
    _expected_calls = advanced_settings["greedy_iterations"] * greedy_tries  # off-graph: once per candidate, NOT per model or recycle
    # Helper: attach BALM + the structure barrier (and optional naturalness) for a
    # discrete stage. The barrier walls off pLDDT/i_ptm < floor so BALM can only move
    # the sequence *laterally* among confidently-folded designs, never into the
    # adversarial low-confidence basin it used to collapse into.
    _balm_cb = {"cb": None}
    _cand_rec = {"rec": None}
    _stage = StageLog()           # (step, stage name) for the animation banner
    _mut_log = MutationLog()      # sampled mutations, for the Stage-4 animation overlay
    _pre_greedy_seq = None        # sequence semigreedy starts from
    _n_frames_designed = None     # frames at end of Stage 4 (excl. replays)
    # --- augment sampling ---------------------------------------------------- #
    _aug = augment_settings(advanced_settings)
    _aug_rec = None               # AugmentationRecorder, only in augment mode
    _aug_selected = []            # candidates handed on to bindcraft_augment.py

    def _attach_guards(af, with_balm):
        """Attach the discrete-stage loss terms.

        The WALLS (structure barrier + hydrophobic fraction) go on whenever
        guards_on, independent of BALM. The BALM affinity term goes on only when
        with_balm. This split is what makes a weight=0 run a valid control: same
        walls, same loss surface, one term removed.
        """
        if with_balm:
            # off-graph: appends to _callbacks["design"]["post"], so the AF2 HLO is
            # byte-identical to the walls-only control run.
            _balm_cb["cb"] = add_balm_design_callback(af, balm_aux, weight=balm_aux.cfg.weight)
        if guards_on:
            add_structure_barrier_loss(af, weight=100.0, plddt_floor=0.8, iptm_floor=0.8)
            add_hydrophobic_fraction_barrier_loss(af, weight=100.0, max_frac=0.85)
            if contact_barrier_on:
                # weight=1.0 on purpose: the schedule values (50/10/5) ARE the increments.
                add_interface_contact_barrier_loss(af, weight=1.0, atom_cutoff=5.0, min_contacts=3)
            # optional (loads a 2nd ESM ~1-3 GB VRAM): keep sequences protein-like too.
            # add_esm_naturalness_loss(af, weight=1.0, device=balm_aux.cfg.device)
        # appended last: design post-callbacks fire in order, so this sees the loss
        # AFTER the BALM callback has adjusted it -- the same scalar semigreedy minimises.
        _cand_rec["rec"] = attach_candidate_recorder(af, top_k=MAX_RUNNERUP_RETRIES + 4)
        print(f"[balm] stage-4 terms: balm={'on' if with_balm else 'OFF'}, "
              f"walls={'on' if guards_on else 'OFF'}, "
              f"contact_wall={'on' if contact_barrier_on else 'OFF'}, "
              f"loss_callbacks={len(af._callbacks['model']['loss'])}")
    ### start design algorithm based on selection
    if advanced_settings["design_algorithm"] == '2stage':
        # uses gradient descend to get a PSSM profile and then uses PSSM to bias the sampling of random mutations to decrease loss
        # WARNING: the BALM term is attached BEFORE the soft PSSM phase here. stop_gradient
        # makes it contribute nothing to the gradient, but the host callback STILL EXECUTES
        # on every soft iteration -- a full 650M ESM pass plus a device stall, for zero
        # effect on the result. The profile report below will show this as a call count far
        # above _expected_calls. If confirmed, split the call into separate soft/hard phases
        # and attach in between (see remove_balm_loss in the bridge).
        _attach_guards(af_model, with_balm=balm_on)
        af_model.design_pssm_semigreedy(soft_iters=advanced_settings["soft_iterations"], hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models,
                                        num_models=stage4_num_models, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)
        stage4_ran = True
        if balm_on:
            _balm_profile_report("2stage", expected=_expected_calls)
    elif advanced_settings["design_algorithm"] == '3stage':
        # 3 stage design using logits, softmax, and one hot encoding
        # NOTE: fully gradient-based -> a non-differentiable BALM scalar would be inert here, so it is NOT attached.
        # BALM has no influence on a 3stage trajectory, so under the BALM-only policy it will be rejected (stage4_ran stays False).
        af_model.design_3stage(soft_iters=advanced_settings["soft_iterations"], temp_iters=advanced_settings["temporary_iterations"], hard_iters=advanced_settings["hard_iterations"],
                                num_models=1, models=design_models, sample_models=advanced_settings["sample_models"], save_best=True)
    elif advanced_settings["design_algorithm"] == 'greedy':
        # design by using random mutations that decrease loss
        _attach_guards(af_model, with_balm=balm_on)
        af_model.design_semigreedy(advanced_settings["greedy_iterations"], tries=greedy_tries, num_models=stage4_num_models, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)
        stage4_ran = True
        if balm_on:
            _balm_profile_report("greedy", expected=_expected_calls)
    elif advanced_settings["design_algorithm"] == 'mcmc':
        # design by using random mutations that decrease loss
        _attach_guards(af_model, with_balm=balm_on)
        half_life = round(advanced_settings["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(advanced_settings["greedy_iterations"], half_life=half_life, T_init=t_mcmc, mutation_rate=greedy_tries, num_models=stage4_num_models, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)
        stage4_ran = True
        if balm_on:
            _balm_profile_report("mcmc", expected=_expected_calls)
    elif advanced_settings["design_algorithm"] == '4stage':
        # initial logits to prescreen trajectory
        print("Stage 1: Test Logits")
        _stage.mark("Stage 1: logits", getattr(af_model, "_k", 0),
                    frame=len(af_model._tmp["traj"]["seq"]))
        af_model.design_logits(iters=50, e_soft=0.9, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)
        # determine pLDDT of best iteration according to lowest 'loss' value
        initial_plddt = get_best_plddt(af_model, length)
        # if best iteration has high enough confidence then continue
        if initial_plddt > 0.65:
            print("Initial trajectory pLDDT good, continuing: "+str(initial_plddt))
            if advanced_settings["optimise_beta"]:
                # temporarily dump model to assess secondary structure
                af_model.save_pdb(model_pdb_path)
                _, beta, *_ = calc_ss_percentage(model_pdb_path, advanced_settings, 'B')
                os.remove(model_pdb_path)
                # if beta sheeted trajectory is detected then choose to optimise
                if float(beta) > 15:
                    advanced_settings["soft_iterations"] = advanced_settings["soft_iterations"] + advanced_settings["optimise_beta_extra_soft"]
                    advanced_settings["temporary_iterations"] = advanced_settings["temporary_iterations"] + advanced_settings["optimise_beta_extra_temp"]
                    af_model.set_opt(num_recycles=advanced_settings["optimise_beta_recycles_design"])
                    print("Beta sheeted trajectory detected, optimising settings")
            # how many logit iterations left
            logits_iter = advanced_settings["soft_iterations"] - 50
            if logits_iter > 0:
                print("Stage 1: Additional Logits Optimisation")
                af_model.clear_best()
                _stage.mark("Stage 1: logits (extra)", getattr(af_model, "_k", 0),
                            frame=len(af_model._tmp["traj"]["seq"]))
                af_model.design_logits(iters=logits_iter, e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
                                    ramp_recycles=False, save_best=True)
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
                print("Optimised logit trajectory pLDDT: "+str(logit_plddt))
            else:
                logit_plddt = initial_plddt
            # perform softmax trajectory design
            if advanced_settings["temporary_iterations"] > 0:
                print("Stage 2: Softmax Optimisation")
                af_model.clear_best()
                _stage.mark("Stage 2: softmax", getattr(af_model, "_k", 0),
                            frame=len(af_model._tmp["traj"]["seq"]))
                af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
                softmax_plddt = get_best_plddt(af_model, length)
            else:
                softmax_plddt = logit_plddt
            # perform one hot encoding
            if softmax_plddt > 0.65:
                print("Softmax trajectory pLDDT good, continuing: "+str(softmax_plddt))
                onehot_plddt = softmax_plddt
                if advanced_settings["hard_iterations"] > 0:
                    af_model.clear_best()
                    print("Stage 3: One-hot Optimisation")
                    _stage.mark("Stage 3: one-hot", getattr(af_model, "_k", 0),
                                frame=len(af_model._tmp["traj"]["seq"]))
                    af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True)
                    onehot_plddt = get_best_plddt(af_model, length)
                if onehot_plddt > 0.65:
                    # perform greedy mutation optimisation
                    print("One-hot trajectory pLDDT good, continuing: "+str(onehot_plddt))
                    # BALM-only policy: the STARTING (Stage-3) design must already clear the barrier
                    # floor, otherwise we do not even start semigreedy -- BALM should steer a
                    # confidently-folded design, not fight up from below the wall.
                    stage4_start_floor = STAGE4_START_PLDDT if guards_on else 0.0
                    if guards_on and onehot_plddt < stage4_start_floor:
                        update_failures(failure_csv, 'Trajectory_stage4_start_pLDDT')
                        print(f"One-hot pLDDT {onehot_plddt} below Stage-4 start floor {stage4_start_floor}; "
                              f"NOT starting semigreedy (trajectory will be rejected).")
                    elif advanced_settings["greedy_iterations"] > 0:
                        print("Stage 4: PSSM Semigreedy Optimisation")
                        # --- attach the BALM affinity term + structure barrier for the discrete Stage 4 only ---
                        af_model.opt["i_con"].update({"cutoff": 10.0})
                        # capture BEFORE clear_best(): aux still holds the Stage-3
                        # result that semigreedy will start mutating from.
                        _pre_greedy_seq = _binder_seq_from_aux(af_model, length)
                        _stage.mark("Stage 4: semigreedy", getattr(af_model, "_k", 0),
                                    frame=len(af_model._tmp["traj"]["seq"]))
                        # reset best BEFORE Stage 4 so the kept design can come ONLY from
                        # the semigreedy window, never an earlier stage. Done whenever the
                        # guards are on -- i.e. in BOTH the BALM and the control run, so
                        # the two are comparable design-for-design.
                        if guards_on:
                            af_model.clear_best()
                        n_before = balm_aux.stats()["predictions"]
                        # profile counters start clean here, so the report below measures
                        # Stage 4 in isolation rather than the whole trajectory.
                        _balm_reset_stats()
                        # BALM term only when balm_on; walls whenever guards_on. See
                        # _attach_guards -- this is the same split, applied inline here
                        # because Stage 4 needs the surrounding bookkeeping.
                        _attach_guards(af_model, with_balm=balm_on)
                        if balm_on:
                            assert _balm_cb["cb"] in af_model._callbacks["design"]["post"], "BALM post-callback not registered!"
                            assert af_model.opt["weights"].get("balm") is None, \
                                "opt['weights']['balm'] is set -- that changes the jitted pytree; the off-graph path must not set it"
                        if guards_on:
                            assert af_model.opt["weights"].get("struct_barrier", 0.0) > 0.0, "structure barrier not registered!"
                        print(f"[balm] Stage 4 setup: balm_weight={balm_aux.cfg.weight if balm_on else 0.0} (off-graph), "
                              f"walls={'on' if guards_on else 'OFF'} (plddt>=0.8,iptm>=0.8,w=100), "
                              f"contact_wall={'on' if contact_barrier_on else 'OFF'}, "
                              f"start_plddt={onehot_plddt}, num_models={stage4_num_models}, "
                              f"loss_callbacks={len(af_model._callbacks['model']['loss'])}")
                        _k0_stage4 = int(getattr(af_model, '_k', 0))
                        # Attached AFTER _attach_guards so its design-post hook is
                        # LAST in the chain and therefore sees the BALM-adjusted
                        # aux["loss"] -- the same scalar semigreedy takes the argmin
                        # of. Attached before the search starts so nothing is missed.
                        if _aug["enabled"]:
                            _aug_rec = AugmentationRecorder(
                                af_model, length, balm_aux=balm_aux,
                                score_balm=_aug["score_balm"],
                                mutation_log=_mut_log).attach().start(_k0_stage4)
                            af_model._mutation_log = _mut_log
                            if _aug["score_balm"] and not balm_on:
                                print("[augment] BALM is OFF: scoring candidates for the CSV "
                                      "costs one ESM pass each (set augment_score_balm=false to skip)")
                        else:
                            # animation overlay only: which mutations were sampled
                            attach_mutation_recorder(af_model, _mut_log)
                        _t_s4 = time.perf_counter()
                        _pre_s4 = _t_s4 - _t_traj      # stages 1-3 + setup
                        af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models,
                                                        num_models=stage4_num_models, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)
                        _s4 = time.perf_counter() - _t_s4
                        print(f"[timing] stages1-3+setup {_pre_s4:.1f}s | stage4 {_s4:.1f}s "
                              f"({_s4/max(_pre_s4+_s4,1e-9)*100:.0f}% of trajectory so far) | "
                              f"{_s4/max(advanced_settings['greedy_iterations']*greedy_tries,1):.2f}s per candidate")
                        if _aug_rec is not None:
                            _aug_rec.stop().mark_accepted()
                            print(f"[augment] Stage 4 recorded: {_aug_rec.summary()}")
                        stage4_ran = True
                        # frame count at the END of the search: any frames appended
                        # after this come from replay_candidate's _save_results and
                        # are re-predictions, not search iterations.
                        _n_frames_designed = len(af_model._tmp["traj"]["seq"])
                        if balm_on:
                            fired = balm_aux.stats()["predictions"] - n_before
                            print(f"[balm] Stage 4 done. BALM scored {fired} sequences this trajectory; stats={balm_aux.stats()}")
                            _balm_profile_report("Stage 4", expected=_expected_calls)
                            assert fired > 0, "BALM attached but never fired during Stage 4 -- check inputs['seq'] extraction / residue order!"
                else:
                    update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    print("One-hot trajectory pLDDT too low to continue: "+str(onehot_plddt))
            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                print("Softmax trajectory pLDDT too low to continue: "+str(softmax_plddt))
        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            print("Initial trajectory pLDDT too low to continue: "+str(initial_plddt))
    else:
        print("ERROR: No valid design model selected")
        exit()
        return
    ### save trajectory PDB
    af_model.save_pdb(model_pdb_path)

    # --- augment sampling: dump every candidate, then pick who goes to MPNN --- #
    if _aug_rec is not None:
        _aug_dir = os.path.dirname(os.path.abspath(design_paths["Trajectory"]))
        _aug_csv = augment_paths(advanced_settings.get("augment_design_path", _aug_dir))
        _aug_records = _aug_rec.deduplicated() if _aug["dedupe"] else _aug_rec.records
        try:
            n = write_augmentation(_aug_csv["augmentation"], _aug_records,
                                   design_name=design_name, length=length, seed=seed,
                                   helicity=helicity_value, balm_on=balm_on)
            print(f"[augment] wrote {n} rows to {os.path.basename(_aug_csv['augmentation'])}")
        except Exception as _e:
            print(f"[augment] could not write Augmentation.csv ({_e})")
        # Selection is over UNIQUE sequences regardless of the dedupe setting --
        # passing the same sequence to MPNN twice would just duplicate the work.
        _aug_selected = select_candidates(_aug_rec.deduplicated(),
                                          max_negative=_aug["max_negative"])
        _sel_balm = [c.balm for c in _aug_selected]
        print(f"[augment] selected {len(_aug_selected)} candidate(s) for MPNN; BALM {_sel_balm}")

    def _assess():
        """Geometry/confidence verdict for whatever design is currently written to
        model_pdb_path. Returns (terminate, reason, plddt, clashes, n_contacts).

        `terminate` is the folder-name string BindCraft moves the PDB into ("" =
        keep). `reason` is the finer-grained cause, used to decide whether a
        runner-up sequence is worth trying.
        """
        plddt = get_best_plddt(af_model, length)
        clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)
        if clashes > 0:
            return "Clashing", "clashes", plddt, clashes, None
        if plddt < 0.7:
            return "LowConfidence", "plddt", plddt, clashes, None
        n_contacts = len(hotspot_residues(model_pdb_path).items())
        if n_contacts < 3:
            return "LowConfidence", "contacts", plddt, clashes, n_contacts
        return "", "ok", plddt, clashes, n_contacts

    terminate, reason, final_plddt, ca_clashes, n_contacts = _assess()
    _fallback_rank = 0            # 0 = the semigreedy winner was kept
    _fallback_reason = ""         # why the winner was rejected, if it was

    # --- runner-up fallback -------------------------------------------------- #
    # The semigreedy winner is chosen on LOSS, which knows nothing about CA clashes
    # and only approximates contacts (i_con is CB-CB at 10A; hotspot_residues is
    # atom-atom at 5A). So the best-scoring sequence can still fail geometry while
    # the 2nd/3rd best would pass. ColabDesign discards runners-up
    # (af/design.py:465-467), hence the recorder. Retry only for geometry failures:
    # a low-pLDDT winner means the whole trajectory is bad, not just this sequence.
    # In augment mode the runner-up fallback is redundant -- select_candidates has
    # already chosen which sequences to carry forward, on BALM rather than on
    # geometry -- and the trajectory verdict must not stop us, because a failing
    # design is exactly what Bad_examples.csv is for.
    if _aug["enabled"] and _aug["ignore_trajectory_verdict"]:
        if terminate:
            print(f"[augment] trajectory verdict '{terminate}' ({reason}) noted but ignored; "
                  f"continuing so the candidates can be evaluated")
        _aug_assess = (terminate, reason)
        terminate, reason = "", "augment"
    else:
        _aug_assess = ("", "")

    _RETRIABLE = ("clashes", "contacts")
    if terminate and reason in _RETRIABLE and _cand_rec["rec"] is not None and not _aug["enabled"]:
        rec = _cand_rec["rec"]
        print(f"[fallback] winner failed ({reason}); {rec.summary()}")
        for attempt in range(1, MAX_RUNNERUP_RETRIES + 1):
            cand = rec.nth(attempt)          # rank 0 is the winner we just rejected
            if cand is None:
                print(f"[fallback] no candidate at rank {attempt}; giving up")
                break
            _fallback_reason = _fallback_reason or reason
            replay_candidate(af_model, cand.seq, model_pdb_path, recorder=rec,
                             models=design_models, num_models=stage4_num_models,
                             num_recycles=advanced_settings["num_recycles_design"])
            terminate, reason, final_plddt, ca_clashes, n_contacts = _assess()
            _fallback_rank = attempt
            print(f"[fallback] rank {attempt} (loss {cand.loss:.2f}): "
                  f"plddt={final_plddt} clashes={ca_clashes} contacts={n_contacts} -> "
                  f"{reason}")
            if not terminate:
                print(f"[fallback] recovered with runner-up rank {attempt}")
                break
            if reason not in _RETRIABLE:
                break                        # e.g. this one is simply low-confidence

    af_model.aux["log"]["terminate"] = terminate
    if terminate == "Clashing":
        update_failures(failure_csv, 'Trajectory_Clashes')
        print("Severe clashes detected, skipping analysis and MPNN optimisation")
        print("")
    elif reason == "plddt":
        update_failures(failure_csv, 'Trajectory_final_pLDDT')
        print("Trajectory starting confidence low, skipping analysis and MPNN optimisation")
        print("")
    elif reason == "contacts":
        update_failures(failure_csv, 'Trajectory_Contacts')
        print("Too few contacts at the interface, skipping analysis and MPNN optimisation")
        print("")
    else:
        # phew, trajectory is okay! We can continue
        print("Trajectory successful, final pLDDT: "+str(final_plddt))
    # BALM-only acceptance policy: force-reject any trajectory whose kept design did NOT
    # come out of the BALM-active semigreedy search (Stage 4 skipped because the start was
    # below the floor, greedy_iterations==0, or a non-semigreedy algorithm was used). We
    # only want peptides BALM actually influenced.
    if guards_on and not stage4_ran and af_model.aux["log"]["terminate"] == "":
        af_model.aux["log"]["terminate"] = "LowConfidence"
        update_failures(failure_csv, 'Trajectory_no_semigreedy')
        print("Rejecting: kept design did not come from BALM semigreedy (Stage 4 did not run).")
        print("")
    # move low quality prediction:
    if af_model.aux["log"]["terminate"] != "":
        shutil.move(model_pdb_path, design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"])
    ### get the sampled sequence for plotting
    af_model.get_seqs()
    # augment mode outputs Augmentation.csv, Bad_examples.csv and the animations,
    # and nothing else -- the plots and the trajectory pickle are suppressed even
    # if the settings ask for them.
    if advanced_settings["save_design_trajectory_plots"] and not _aug["enabled"]:
        plot_trajectory(af_model, design_name, design_paths)
    ### save the hallucination trajectory animation
    if advanced_settings["save_design_animations"]:
        _t_anim = time.perf_counter()
        try:
            plots = animate_with_stages(af_model, stage_log=_stage, dpi=150,
                                        e=_n_frames_designed)
        except Exception as _e:
            print(f"[anim] custom animation failed ({_e}); falling back to ColabDesign's")
            plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name+".html"), 'w') as f:
            f.write(plots)
        plt.close('all')
        print(f"[timing] animate {time.perf_counter()-_t_anim:.1f}s")
    if advanced_settings["save_trajectory_pickle"] and not _aug["enabled"]:
        with open(os.path.join(design_paths["Trajectory/Pickle"], design_name+".pickle"), 'wb') as handle:
            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    # XLA compiles for this whole trajectory. Expect a handful on the first one and
    # 0 thereafter; a steady nonzero count means the persistent cache is not hitting.
    # --- provenance for mpnn_design_stats.csv ------------------------------- #
    _rec = _cand_rec["rec"]
    _chosen = _rec.nth(_fallback_rank) if _rec is not None else None
    _traj_seq = _binder_seq_from_best(af_model, length)
    _balm_traj = None
    if _traj_seq and balm_on:
        try:
            _balm_traj = round(float(balm_aux.raw_score(_traj_seq)), 4)   # cached
        except Exception as _e:
            print(f"[balm] could not score trajectory sequence: {_e}")
    af_model._balm_meta = {
        "trajectory_sequence": _traj_seq,
        "balm_trajectory": _balm_traj,
        "greedy_iteration": _rec.iteration_of(_chosen, _k0_stage4) if _rec is not None else None,
        "candidate_rank": _fallback_rank,
        "is_fallback": bool(_fallback_rank),
        "fallback_reason": _fallback_reason,
        "pre_greedy_sequence": _pre_greedy_seq,
        "greedy_step_sequences": ";".join(
            _seq_from_indices(c[1], length)
            for c in (_rec.accepted_trajectory(_k0_stage4) if _rec is not None else [])
        ) or None,
    }
    print(f"[balm] meta: {af_model._balm_meta}")
    # what bindcraft_augment.py picks up: the selected candidates and the context
    # needed to write their Bad_examples.csv rows.
    af_model._augment = {
        "enabled": _aug["enabled"],
        "settings": _aug,
        "selected": _aug_selected,
        "n_recorded": 0 if _aug_rec is None else len(_aug_rec.records),
        "trajectory_verdict": _aug_assess if _aug["enabled"] else ("", ""),
        "design_name": design_name, "length": length, "seed": seed,
        "helicity": helicity_value, "balm_on": balm_on,
    }
    print(f"[timing] trajectory total {time.perf_counter()-_t_traj:.1f}s")
    _compile_report(f"trajectory {design_name}")
    return af_model
# run prediction for binder with masked template target
def predict_binder_complex(prediction_model, binder_sequence, mpnn_design_name, target_pdb, chain, length, trajectory_pdb, prediction_models, advanced_settings, filters, design_paths, failure_csv, seed=None):
    prediction_stats = {}
    # clean sequence
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    # reset filtering conditionals
    pass_af2_filters = True
    filter_failures = {}
    # start prediction per AF2 model, 2 are used by default due to masked templates
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(complex_pdb):
            # predict model
            prediction_model.predict(seq=binder_sequence, models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(complex_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae
            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2),
                'pTM': round(prediction_metrics['ptm'], 2),
                'i_pTM': round(prediction_metrics['i_ptm'], 2),
                'pAE': round(prediction_metrics['pae'], 2),
                'i_pAE': round(prediction_metrics['i_pae'], 2)
            }
            prediction_stats[model_num+1] = stats
            # List of filter conditions and corresponding keys
            filter_conditions = [
                (f"{model_num+1}_pLDDT", 'plddt', '>='),
                (f"{model_num+1}_pTM", 'ptm', '>='),
                (f"{model_num+1}_i_pTM", 'i_ptm', '>='),
                (f"{model_num+1}_pAE", 'pae', '<='),
                (f"{model_num+1}_i_pAE", 'i_pae', '<='),
            ]
            # perform initial AF2 values filtering to determine whether to skip relaxation and interface scoring
            for filter_name, metric_key, comparison in filter_conditions:
                threshold = filters.get(filter_name, {}).get("threshold")
                if threshold is not None:
                    if comparison == '>=' and prediction_metrics[metric_key] < threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
                    elif comparison == '<=' and prediction_metrics[metric_key] > threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
            if not pass_af2_filters:
                break
    # Update the CSV file with the failure counts
    if filter_failures:
        update_failures(failure_csv, filter_failures)
    # AF2 filters passed, contuing with relaxation
    for model_num in prediction_models:
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if pass_af2_filters:
            mpnn_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            pr_relax(complex_pdb, mpnn_relaxed)
        else:
            if os.path.exists(complex_pdb):
                os.remove(complex_pdb)
    return prediction_stats, pass_af2_filters
# run prediction for binder alone
def predict_binder_alone(prediction_model, binder_sequence, mpnn_design_name, length, trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths, seed=None):
    binder_stats = {}
    # prepare sequence for prediction
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    prediction_model.set_seq(binder_sequence)
    # predict each model separately
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        binder_alone_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(binder_alone_pdb):
            # predict model
            prediction_model.predict(models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(binder_alone_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, pae
            # align binder model to trajectory binder
            try:
                align_pdbs(trajectory_pdb, binder_alone_pdb, binder_chain, "A")
            except Exception as e:
                print(f"Warning: alignment failed for {mpnn_design_name}_model{model_num+1}: {e}")
            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2),
                'pTM': round(prediction_metrics['ptm'], 2),
                'pAE': round(prediction_metrics['pae'], 2)
            }
            binder_stats[model_num+1] = stats
    return binder_stats
# run MPNN to generate sequences for binders
def mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings):
    # clear GPU memory
    clear_mem()
    # initialise MPNN model
    mpnn_model = mk_mpnn_model(backbone_noise=advanced_settings["backbone_noise"], model_name=advanced_settings["model_path"], weights=advanced_settings["mpnn_weights"])
    # check whether keep the interface generated by the trajectory or whether to redesign with MPNN
    design_chains = 'A,' + binder_chain
    if advanced_settings["mpnn_fix_interface"]:
        fixed_positions = 'A,' + trajectory_interface_residues
        fixed_positions = fixed_positions.rstrip(",")
        print("Fixing interface residues: "+trajectory_interface_residues)
    else:
        fixed_positions = 'A'
    # prepare inputs for MPNN
    mpnn_model.prep_inputs(pdb_filename=trajectory_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=advanced_settings["omit_AAs"])
    # sample MPNN sequences in parallel
    mpnn_sequences = mpnn_model.sample(temperature=advanced_settings["sampling_temp"], num=1, batch=advanced_settings["num_seqs"])
    return mpnn_sequences
# Get pLDDT of best model
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)
# Define radius of gyration loss for colabdesign
def add_rg_loss(self, weight=0.1):
    '''add radius of gyration loss'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365
        rg = jax.nn.elu(rg - rg_th)
        return {"rg":rg}
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight
# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}
    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight
# add helicity loss
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):
      if "offset" in inputs:
        offset = inputs["offset"]
      else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:,None] - idx[None,:]
      # define distogram
      dgram = outputs["distogram"]["logits"]
      dgram_bins = get_dgram_bins(outputs)
      mask_2d = np.outer(np.append(np.zeros(self._target_len), np.ones(self._binder_len)), np.append(np.zeros(self._target_len), np.ones(self._binder_len)))
      x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
      if offset is None:
        if mask_2d is None:
          helix_loss = jnp.diagonal(x,3).mean()
        else:
          helix_loss = jnp.diagonal(x * mask_2d,3).sum() + (jnp.diagonal(mask_2d,3).sum() + 1e-8)
      else:
        mask = offset == 3
        if mask_2d is not None:
          mask = jnp.where(mask_2d,mask,0)
        helix_loss = jnp.where(mask,x,0.0).sum() / (mask.sum() + 1e-8)
      return {"helix":helix_loss}
    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight
# add N- and C-terminus distance loss
def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    """Add loss penalizing the distance between N and C termini"""
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # Considering only the last _binder_len residues
        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]
        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)
        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)
        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}
    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight
# plot design trajectory losses
def plot_trajectory(af_model, design_name, design_paths):
    # NOTE: 'hphob_iface' and 'hphob_frac' were previously sitting in the COLORS list
    # rather than here, which meant matplotlib received e.g. color='hphob_iface' and
    # raised ValueError for any trajectory logging >=8 metrics. They are metrics; the
    # colors list is now actual colors only.
    metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'rg', 'mpnn',
                       'balm', 'struct_barrier', 'naturalness', 'hphob_iface', 'hphob_frac']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            # Create a new figure for each metric
            plt.figure()
            loss = af_model.get_loss(metric)
            # Create an x axis for iterations
            iterations = range(1, len(loss) + 1)
            plt.plot(iterations, loss, label=f'{metric}', color=colors[index % len(colors)])
            # Add labels and a legend
            plt.xlabel('Iterations')
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)
            # Save the plot
            plt.savefig(os.path.join(design_paths["Trajectory/Plots"], design_name+"_"+metric+".png"), dpi=150)
            # Close the figure
            plt.close()