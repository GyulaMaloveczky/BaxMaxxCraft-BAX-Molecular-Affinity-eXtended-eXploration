####################################
###### BindCraft -- Augment sampling
####################################
"""Data-collection loop.

Same trajectory as bindcraft.py, different purpose: instead of keeping the one
sequence semigreedy liked best, keep everything it looked at, then push a small
BALM-chosen shortlist through MPNN and record the ones that fail.

OUTPUTS (and nothing else, by design)
    Augmentation.csv    every sequence Stage 4 evaluated, with the scores that
                        evaluation already produced. Written by
                        binder_hallucination via augment_sampling.py.
    Bad_examples.csv    one row per shortlisted candidate whose MPNN redesigns
                        all failed the filters.
    Trajectory/Animation/*.html
    failure_csv.csv     not a deliverable -- predict_binder_complex writes to it
                        and would crash without it. Ignore or delete it.

STOPPING
    The loop runs until Bad_examples.csv holds `number_of_final_designs` rows
    (override with advanced_settings["augment_target_bad_examples"]), or until
    max_trajectories is hit.

RELATIONSHIP TO bindcraft.py
    The per-MPNN-sequence scoring block below is a transcription of
    bindcraft.py's, minus the CSV writes and the Accepted/Rejected bookkeeping.
    It has to build `mpnn_data` in exactly design_labels order because
    check_filters() indexes it positionally -- so if you change that block in
    bindcraft.py, change it here too. That duplication is deliberate: importing
    it would mean refactoring the main loop, which this mode is not allowed to
    disturb.
"""
import os
import jax

cache_dir = "/tmp/jax_cache"
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from functions import *
from functions.colabdesign_utils import get_balm_scorer
from functions.augment_sampling import (
    augment_settings, augment_paths, augment_target_reached, count_bad_examples,
    is_bad_example, bad_example_row, append_rows, replay_sequence,
    af2_prefilter_failures,
)

check_jax_gpu()

######################################
### parse input paths
parser = argparse.ArgumentParser(description='BindCraft augment-sampling data collection.')
parser.add_argument('--settings', '-s', type=str, required=True,
                    help='Path to the basic settings.json file. Required.')
parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                    help='Path to the filters.json file used to filter designs.')
parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer.json',
                    help='Path to the advanced.json file with additional design settings.')

args = parser.parse_args()
settings_path, filters_path, advanced_path = perform_input_check(args)
target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)

settings_file = os.path.basename(settings_path).split('.')[0]
filters_file = os.path.basename(filters_path).split('.')[0]
advanced_file = os.path.basename(advanced_path).split('.')[0]

design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])

bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

# this script IS the mode; do not make the user remember to turn it on
advanced_settings["augment_sampling"] = True
aug = augment_settings(advanced_settings)

design_paths = generate_directories(target_settings["design_path"])
trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

paths = augment_paths(target_settings["design_path"])
failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')
generate_filter_pass_csv(failure_csv, args.filters)

# how many bad examples we are after
target_bad = advanced_settings.get("augment_target_bad_examples") or target_settings["number_of_final_designs"]

pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
print(f"Augment sampling for target {settings_file}")
print(f"Design settings: {advanced_file} | filters: {filters_file}")
print(f"Bad-example rule: a candidate is bad if it fails anything "
      f"{'except' if aug['mode'] == 'exclude' else 'in'} {list(aug['ignored'])}")
print(f"Collecting until Bad_examples.csv has {target_bad} rows (currently {count_bad_examples(paths['bad_examples'])})")

binder_chain = "B"
script_start_time = time.time()
trajectory_n = 1


def prepare_candidate(candidate, rank, candidate_pdb, length, design_name):
    """Phase A: relax the candidate, find its interface, and run MPNN on it.

    This is the ONLY part that calls mpnn_gen_sequence, and mpnn_gen_sequence
    calls clear_mem() on entry (colabdesign_utils line 676) -- it wipes EVERY
    live JAX buffer on the GPU. So it must run BEFORE the prediction models are
    built, never interleaved with them. That was the bug behind the persistent
    "Array has been deleted with shape=float32[64]": the prediction models were
    built first, then mpnn_gen_sequence inside the eval loop deleted their
    buffers, and predict_binder_alone touched a freed monomer input.

    Returns a dict of everything phase B needs, or None if the candidate has no
    interface / no MPNN sequences survive.
    """
    cand_name = f"{design_name}_c{rank}"

    # interface residues of THIS candidate (MPNN fixes them), from the relaxed pose
    candidate_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], cand_name + ".pdb")
    pr_relax(candidate_pdb, candidate_relaxed)
    _scores, _aa, interface_residues = score_interface(candidate_relaxed, binder_chain)
    if not interface_residues:
        print(f"[augment] {cand_name}: no interface residues, cannot run MPNN")
        return None

    mpnn_trajectories = mpnn_gen_sequence(candidate_pdb, binder_chain, interface_residues, advanced_settings)
    restricted_AAs = set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(',')) if advanced_settings["force_reject_AA"] else set()
    mpnn_sequences = sorted({
        mpnn_trajectories['seq'][n][-length:]: {
            'seq': mpnn_trajectories['seq'][n][-length:],
            'score': mpnn_trajectories['score'][n],
            'seqid': mpnn_trajectories['seqid'][n]
        } for n in range(advanced_settings["num_seqs"])
        if (not restricted_AAs or not any(aa in mpnn_trajectories['seq'][n][-length:].upper() for aa in restricted_AAs))
    }.values(), key=lambda x: x['score'])

    if not mpnn_sequences:
        print(f"[augment] {cand_name}: no MPNN sequences survived the AA restrictions")
        return None

    # A candidate is a pass the moment ONE redesign clears every filter, so there
    # is nothing to learn from the rest -- stop there. And cap the rest: num_seqs
    # is 50 by default, which at ~5 candidates a trajectory would be 250 AF2
    # predictions before a single row is written.
    max_mpnn = int(advanced_settings.get("augment_mpnn_per_candidate", 5))
    mpnn_sequences = mpnn_sequences[:max_mpnn]

    return {"candidate": candidate, "rank": rank, "cand_name": cand_name,
            "candidate_pdb": candidate_pdb, "interface_residues": interface_residues,
            "mpnn_sequences": mpnn_sequences, "length": length}


def score_candidate(prep, seed, helicity_value, design_name, complex_model, binder_model):
    """Phase B: AF2-predict and filter the MPNN sequences from prepare_candidate().

    No clear_mem() reachable from here -- neither predict_binder_complex nor
    predict_binder_alone clears -- so the two models built in phase B stay valid
    across every candidate.

    Returns (n_mpnn, passed, results) where results is
    [(mpnn_sequence, True | [failed filter columns]), ...].
    """
    candidate = prep["candidate"]
    rank = prep["rank"]
    cand_name = prep["cand_name"]
    candidate_pdb = prep["candidate_pdb"]
    interface_residues = prep["interface_residues"]
    mpnn_sequences = prep["mpnn_sequences"]
    length = prep["length"]

    # prep the prediction models against THIS candidate's structure
    if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
        complex_model.prep_inputs(pdb_filename=candidate_pdb, chain='A', binder_chain='B', binder_len=length,
                                  use_binder_template=True, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                  rm_target_sc=advanced_settings["rm_template_sc_predict"], rm_template_ic=True)
    else:
        complex_model.prep_inputs(pdb_filename=target_settings["starting_pdb"], chain=target_settings["chains"],
                                  binder_len=length, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                  rm_target_sc=advanced_settings["rm_template_sc_predict"])

    results = []
    mpnn_n = 1
    for mpnn_sequence in mpnn_sequences:
        mpnn_time = time.time()
        mpnn_design_name = cand_name + "_mpnn" + str(mpnn_n)
        mpnn_score = round(mpnn_sequence['score'], 2)
        mpnn_seqid = round(mpnn_sequence['seqid'], 2)

        mpnn_complex_statistics, pass_af2_filters = predict_binder_complex(
            complex_model, mpnn_sequence['seq'], mpnn_design_name,
            target_settings["starting_pdb"], target_settings["chains"],
            length, candidate_pdb, prediction_models, advanced_settings,
            filters, design_paths, failure_csv)

        if not pass_af2_filters:
            # predict_binder_complex keeps its failure list to itself, so re-derive
            # it from the stats it did return -- same comparisons, same names.
            failed = af2_prefilter_failures(mpnn_complex_statistics, filters)
            print(f"[augment] {mpnn_design_name}: AF2 prefilters failed {failed}")
            results.append((mpnn_sequence['seq'], failed or ["AF2_prefilter"]))
            mpnn_n += 1
            continue

        for model_num in prediction_models:
            mpnn_design_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            mpnn_design_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            if os.path.exists(mpnn_design_pdb):
                num_clashes_mpnn = calculate_clash_score(mpnn_design_pdb)
                num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)
                mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(mpnn_design_relaxed, binder_chain)
                mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface, mpnn_beta_interface, mpnn_loops_interface, mpnn_i_plddt, mpnn_ss_plddt = calc_ss_percentage(mpnn_design_pdb, advanced_settings, binder_chain)
                rmsd_site = unaligned_rmsd(candidate_pdb, mpnn_design_pdb, binder_chain, binder_chain)
                target_rmsd = target_pdb_rmsd(mpnn_design_pdb, target_settings["starting_pdb"], target_settings["chains"])

                mpnn_complex_statistics[model_num+1].update({
                    'i_pLDDT': mpnn_i_plddt,
                    'ss_pLDDT': mpnn_ss_plddt,
                    'Unrelaxed_Clashes': num_clashes_mpnn,
                    'Relaxed_Clashes': num_clashes_mpnn_relaxed,
                    'Binder_Energy_Score': mpnn_interface_scores['binder_score'],
                    'Surface_Hydrophobicity': mpnn_interface_scores['surface_hydrophobicity'],
                    'ShapeComplementarity': mpnn_interface_scores['interface_sc'],
                    'PackStat': mpnn_interface_scores['interface_packstat'],
                    'dG': mpnn_interface_scores['interface_dG'],
                    'dSASA': mpnn_interface_scores['interface_dSASA'],
                    'dG/dSASA': mpnn_interface_scores['interface_dG_SASA_ratio'],
                    'Interface_SASA_%': mpnn_interface_scores['interface_fraction'],
                    'Interface_Hydrophobicity': mpnn_interface_scores['interface_hydrophobicity'],
                    'n_InterfaceResidues': mpnn_interface_scores['interface_nres'],
                    'n_InterfaceHbonds': mpnn_interface_scores['interface_interface_hbonds'],
                    'InterfaceHbondsPercentage': mpnn_interface_scores['interface_hbond_percentage'],
                    'n_InterfaceUnsatHbonds': mpnn_interface_scores['interface_delta_unsat_hbonds'],
                    'InterfaceUnsatHbondsPercentage': mpnn_interface_scores['interface_delta_unsat_hbonds_percentage'],
                    'InterfaceAAs': mpnn_interface_AA,
                    'Interface_Helix%': mpnn_alpha_interface,
                    'Interface_BetaSheet%': mpnn_beta_interface,
                    'Interface_Loop%': mpnn_loops_interface,
                    'Binder_Helix%': mpnn_alpha,
                    'Binder_BetaSheet%': mpnn_beta,
                    'Binder_Loop%': mpnn_loops,
                    'Hotspot_RMSD': rmsd_site,
                    'Target_RMSD': target_rmsd
                })
                if advanced_settings["remove_unrelaxed_complex"]:
                    os.remove(mpnn_design_pdb)

        mpnn_complex_averages = calculate_averages(mpnn_complex_statistics, handle_aa=True)

        binder_statistics = predict_binder_alone(binder_model, mpnn_sequence['seq'], mpnn_design_name, length,
                                                 candidate_pdb, binder_chain, prediction_models, advanced_settings, design_paths)
        for model_num in prediction_models:
            mpnn_binder_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            rmsd_binder = None
            if os.path.exists(mpnn_binder_pdb):
                try:
                    rmsd_binder = unaligned_rmsd(candidate_pdb, mpnn_binder_pdb, binder_chain, "A", superimpose=True)
                except Exception as e:
                    print(f"Warning: failed to calculate binder RMSD for {mpnn_design_name}_model{model_num+1}: {e}")
            binder_statistics[model_num+1].update({'Binder_RMSD': rmsd_binder})
            if advanced_settings["remove_binder_monomer"] and os.path.exists(mpnn_binder_pdb):
                os.remove(mpnn_binder_pdb)
        binder_averages = calculate_averages(binder_statistics)

        seq_notes = validate_design_sequence(mpnn_sequence['seq'], mpnn_complex_averages.get('Relaxed_Clashes', None), advanced_settings)
        mpnn_end_time = time.time() - mpnn_time
        elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(mpnn_end_time // 3600), int((mpnn_end_time % 3600) // 60), int(mpnn_end_time % 60))}"

        # ---- build mpnn_data in design_labels order; check_filters indexes it ---- #
        model_numbers = range(1, 6)
        statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                             'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                             'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                             'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

        mpnn_data = [mpnn_design_name, advanced_settings["design_algorithm"], length, seed, helicity_value,
                     target_settings["target_hotspot_residues"], mpnn_sequence['seq'], mpnn_interface_residues, mpnn_score, mpnn_seqid]
        for label in statistics_labels:
            mpnn_data.append(mpnn_complex_averages.get(label, None))
            for model in model_numbers:
                mpnn_data.append(mpnn_complex_statistics.get(model, {}).get(label, None))
        for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:
            mpnn_data.append(binder_averages.get(label, None))
            for model in model_numbers:
                mpnn_data.append(binder_statistics.get(model, {}).get(label, None))
        mpnn_data.extend([elapsed_mpnn_text, seq_notes, settings_file, filters_file, advanced_file])

        _balm_mpnn = None
        _scorer = get_balm_scorer()
        if _scorer is not None:
            try:
                _balm_mpnn = round(float(_scorer.raw_score(mpnn_sequence['seq'])), 4)
            except Exception as _e:
                print(f"[balm] MPNN rescore failed for {mpnn_design_name}: {_e}")
        mpnn_data.extend([
            candidate.sequence, interface_residues, candidate.balm, _balm_mpnn,
            candidate.iteration, rank, bool(rank), "augment",
            candidate.sequence, None,
        ])

        filter_conditions = check_filters(mpnn_data, design_labels, filters)
        if filter_conditions is True:
            print(f"[augment] {mpnn_design_name} passed all filters -- candidate is a PASS")
            results.append((mpnn_sequence['seq'], True))
            break
        print(f"[augment] {mpnn_design_name} failed {list(filter_conditions)}")
        results.append((mpnn_sequence['seq'], list(filter_conditions)))
        mpnn_n += 1

    passed = any(v is True for _s, v in results)
    return len(results), passed, results


### start collection loop
while True:
    done, n_bad = augment_target_reached(paths["bad_examples"], target_bad)
    if done:
        print(f"Collected {n_bad} bad examples, target reached.")
        break

    if check_n_trajectories(design_paths, advanced_settings):
        print("Maximum trajectories reached.")
        break

    trajectory_start_time = time.time()
    seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
    samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
    length = np.random.choice(samples)
    helicity_value = load_helicity(advanced_settings)

    design_name = target_settings["binder_name"] + "_l" + str(length) + "_s" + str(seed)
    trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
    if any(os.path.exists(os.path.join(design_paths[d], design_name + ".pdb")) for d in trajectory_dirs):
        continue

    print(f"Starting trajectory: {design_name}   [{n_bad}/{target_bad} bad examples]")
    trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"],
                                      target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                      design_models, advanced_settings, design_paths, failure_csv)

    state = getattr(trajectory, "_augment", {}) or {}
    selected = state.get("selected") or []
    trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
    print(f"[augment] {state.get('n_recorded', 0)} candidates recorded, {len(selected)} selected")

    if not selected:
        trajectory_n += 1
        gc.collect()
        continue

    # ================================================================== #
    # THE ORDERING RULE (this is what the float32[64] crash was about):
    #
    #   Three different GPU phases each call clear_mem() on entry, and
    #   clear_mem() deletes EVERY live JAX buffer on the device, not just one
    #   model's (colabdesign.shared.utils). So the phases can never overlap:
    #     * replay        -- uses the live hallucination model
    #     * mpnn_gen_seq  -- clears, builds an MPNN model  (colabdesign_utils:676)
    #     * AF2 predict   -- clears, builds complex+binder models
    #   The previous version built the prediction models and THEN called
    #   mpnn_gen_sequence inside the eval loop; that clear wiped the just-built
    #   models, and predict_binder_alone then touched a freed monomer input.
    #
    #   Fix: do ALL clear_mem-heavy prep up front (Phase A: replay + relax +
    #   interface + MPNN), then build the prediction models ONCE and only
    #   predict (Phase B). Nothing in Phase B calls clear_mem, so the models
    #   stay valid across every candidate.
    # ================================================================== #

    # --- Phase A1: replay every shortlisted candidate to a PDB (needs the live
    # hallucination model, which is still on device from binder_hallucination).
    candidate_pdbs = []
    for rank, cand in enumerate(selected):
        pdb = os.path.join(design_paths["Trajectory"], f"{design_name}_c{rank}.pdb")
        try:
            replay_sequence(trajectory, cand.sequence, pdb, models=design_models,
                            num_models=1, num_recycles=advanced_settings["num_recycles_design"])
            candidate_pdbs.append((rank, cand, pdb))
        except Exception as e:
            print(f"[augment] could not replay candidate {rank} ({e})")

    # Drop references into the hallucination model before the MPNN clear_mem, so
    # its buffers are genuinely dead and nothing dangles.
    trajectory = None
    del selected
    gc.collect()

    # --- Phase A2: relax + interface + MPNN for every candidate. Each
    # prepare_candidate() calls mpnn_gen_sequence(), which clears the GPU, so
    # this MUST finish before any prediction model exists. Results are plain
    # Python (sequences + interface strings) and survive all later clears.
    preps = []
    for rank, cand, pdb in candidate_pdbs:
        try:
            prep = prepare_candidate(cand, rank, pdb, length, design_name)
        except Exception as e:
            print(f"[augment] candidate {rank} preparation failed ({e})")
            prep = None
        if prep is not None:
            preps.append(prep)

    if not preps:
        print("[augment] no candidates produced MPNN sequences; skipping trajectory")
        trajectory_n += 1
        gc.collect()
        continue

    # --- Phase B: MPNN is done, so it is finally safe to build the prediction
    # models. clear_mem() here frees the MPNN model; from this point on nothing
    # calls clear_mem again, so complex_model/binder_model stay valid for every
    # candidate. Same binder length throughout, so per-candidate prep_inputs
    # reuses the compiled graph rather than recompiling.
    clear_mem()
    complex_model = mk_afdesign_model(protocol="binder", num_recycles=advanced_settings["num_recycles_validation"],
                                      data_dir=advanced_settings["af_params_dir"], use_multimer=multimer_validation,
                                      use_initial_guess=advanced_settings["predict_initial_guess"],
                                      use_initial_atom_pos=advanced_settings["predict_bigbang"])
    binder_model = mk_afdesign_model(protocol="hallucination", use_templates=False, initial_guess=False,
                                     use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"],
                                     data_dir=advanced_settings["af_params_dir"], use_multimer=multimer_validation)
    binder_model.prep_inputs(length=length)

    for prep in preps:
        rank = prep["rank"]
        cand = prep["candidate"]
        try:
            n_mpnn, passed, results = score_candidate(
                prep, seed, helicity_value, design_name, complex_model, binder_model)
        except Exception as e:
            print(f"[augment] candidate {rank} scoring failed ({e})")
            continue

        bad, counted, all_failed = is_bad_example(results, aug["ignored"], aug["mode"])
        if passed:
            print(f"[augment] candidate {rank} PASSES (an MPNN design cleared every filter)")
            continue
        if not bad:
            print(f"[augment] candidate {rank} failed only ignored filters {all_failed}; not recorded")
            continue

        row = bad_example_row(cand, results, counted, all_failed,
                              design_name=design_name, candidate_rank=rank,
                              length=length, seed=seed, helicity=helicity_value,
                              trajectory_verdict=state.get("trajectory_verdict", ("", ""))[0],
                              balm_on=state.get("balm_on"),
                              settings=settings_file, filters_file=filters_file,
                              advanced=advanced_file)
        append_rows(paths["bad_examples"], [row])
        print(f"[augment] candidate {rank} -> Bad_examples.csv (failed {counted})")

    if advanced_settings["remove_unrelaxed_trajectory"] and os.path.exists(trajectory_pdb):
        os.remove(trajectory_pdb)

    t = time.time() - trajectory_start_time
    print(f"Trajectory {design_name} took {int(t // 60)}m {int(t % 60)}s")
    trajectory_n += 1
    gc.collect()

elapsed_time = time.time() - script_start_time
print(f"Finished. {trajectory_n - 1} trajectories, "
      f"{count_bad_examples(paths['bad_examples'])} bad examples, "
      f"{int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m")