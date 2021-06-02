import multiprocessing
import os
import pathlib
import subprocess
import time
import typing

import logging

import psutil
import retry

from stent_opt.struct_opt import design
from stent_opt.struct_opt import construct_model
from stent_opt.struct_opt.design import StentParams
from stent_opt.struct_opt import generation, optimisation_parameters
from stent_opt.struct_opt import patch_manager
# from stent_opt.struct_opt import generation_FORCE, chains

from stent_opt.struct_opt import history
from stent_opt.struct_opt.computer import this_computer

logging.basicConfig()

working_dir_orig = os.getcwd()
working_dir_extract = os.path.join(working_dir_orig, "odb_interface")

# This seems to be required to get
try:
    os.environ.pop('PYTHONIOENCODING')
except KeyError:
    pass


if multiprocessing.parent_process() is None:
    mp_lock = multiprocessing.Lock()

# TODO 2021-02-03
# - Graphs of volume ratios and target volume ratio
# - Rasterise and build (offline) the images of each step.


def make_full_model_list(stent_design: design.StentDesign) -> typing.List[patch_manager.FullModelInfo]:
    return [patch_manager.FullModelInfo(stent_design=stent_design)]


def kill_process_id(proc_id: int):
    process = psutil.Process(proc_id)
    for proc in process.children(recursive=True):
        proc.kill()

    process.kill()


@retry.retry(tries=1000, delay=2, )
def _run_external_command(path, args):

    t_start = time.time()

    old_working_dir = os.getcwd()

    os.chdir(path)

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    TIMEOUT = 3600 * 24  # 1 day

    out, errs = [], []
    try:
        out, errs = proc.communicate(timeout=TIMEOUT)
        ret_code = proc.returncode

    except subprocess.TimeoutExpired:
        ret_code = 'TimeoutExpired on {0}'.format(args)

        try:
            kill_process_id(proc.pid)

        except psutil.NoSuchProcess:
            # No problem if it's already gone in the meantime...
            pass

    if ret_code:
        print(path)
        print(out.decode())
        print(errs.decode())
        raise subprocess.SubprocessError(ret_code)

    os.chdir(old_working_dir)

    t_end = time.time()
    t_elapsed = t_end - t_start
    print(f"  Took {t_elapsed} seconds to {' '.join(args)}")


def run_model(optim_params, inp_fn, force_single_core: bool):

    path, fn = os.path.split(inp_fn)
    fn_solo = os.path.splitext(fn)[0]
    #print(multiprocessing.current_process().name, fn_solo)
    if force_single_core:
        n_cpus = 1
    else:
        n_cpus = this_computer.n_cpus_abaqus_explicit if optim_params.is_explicit else this_computer.n_cpus_abaqus_implicit

    args = ['abaqus.bat', f'cpus={n_cpus}', f'job={fn_solo}']

    # This seems to need to come right after "job" in the argument list
    if optim_params.use_double_precision:
        args.append('double')

    args.extend(["ask_delete=OFF", 'interactive'])

    _run_external_command(path, args)


def perform_extraction(odb_fn, out_db_fn, override_z_val, working_dir_extract, add_initial_node_pos):

    args = ["abaqus.bat", "cae", "noGui=odb_extract.py", "--", str(odb_fn), str(out_db_fn), str(override_z_val), str(add_initial_node_pos)]

    _run_external_command(working_dir_extract, args)



def _from_scract_setup(working_dir, mp_lock) -> generation.RunOneArgs:
    init(mp_lock)
    history_db_fn = history.make_history_db(working_dir)

    with history.History(history_db_fn) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()

    # Do the initial setup from a first model.
    starting_i = 0
    fn_inp = history.make_fn_in_dir(working_dir, ".inp", starting_i)
    current_design = design.make_initial_design(stent_params)
    construct_model.make_stent_model(optim_params, current_design, make_full_model_list(current_design), fn_inp)

    run_model(optim_params, fn_inp, force_single_core=False)
    perform_extraction(
        history.make_fn_in_dir(working_dir, ".odb", starting_i),
        history.make_fn_in_dir(working_dir, ".db", starting_i),
        current_design.stent_params.nodal_z_override_in_odb,
        working_dir_extract,
        add_initial_node_pos=optim_params.add_initial_node_pos,
    )

    with history.History(history_db_fn) as hist:
        elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(stent_params.divs)}
        active_elem_nums = (elem_indices_to_num[idx] for idx in current_design.active_elements)
        snapshot = history.Snapshot(
            iteration_num=starting_i,
            label=f"First Iteration {starting_i}",
            filename=str(fn_inp),
            active_elements=frozenset(active_elem_nums),
        )

        hist.add_snapshot(snapshot)

    # Run the patch models to complete the sensitivity.
    patch_suffix_to_submod_infos = generation.produce_patch_models(working_dir, 0)
    all_run_one_args_in = []
    child_patch_run_one_args = []
    for patch_suffix, model_infos in patch_suffix_to_submod_infos.items():
        run_args_patch = generation.RunOneArgs(
            working_dir=working_dir,
            optim_params=optim_params,
            iter_this=0,
            nodal_z_override_in_odb=stent_params.nodal_z_override_in_odb,
            working_dir_extract=working_dir_extract,
            model_infos=model_infos,
            patch_suffix=patch_suffix,
            child_patch_run_one_args=tuple(),
            executed_feedback_text="",
            do_run_model_TESTING=True,
            do_run_extraction_TESTING=True,
        )
        all_run_one_args_in.append(run_args_patch)

    # Can fork a pool here if needs be...
    if this_computer.n_abaqus_parallel_solves > 1:
        with multiprocessing.Pool(processes=this_computer.n_abaqus_parallel_solves, initializer=init, initargs=(mp_lock,)) as pool:
            for out_run_one_args in pool.imap_unordered(process_pool_run_and_process, all_run_one_args_in):
                print(f"   "+out_run_one_args.executed_feedback_text)
                child_patch_run_one_args.append(out_run_one_args)

    else:
        init(mp_lock)
        for run_args_patch in all_run_one_args_in:
            out_run_one_args = process_pool_run_and_process(run_args_patch)
            print(f"   "+out_run_one_args.executed_feedback_text)
            child_patch_run_one_args.append(out_run_one_args)

    return generation.RunOneArgs(
        working_dir=working_dir,
        optim_params=optim_params,
        iter_this=0,
        nodal_z_override_in_odb=stent_params.nodal_z_override_in_odb,
        working_dir_extract=working_dir_extract,
        model_infos=make_full_model_list(current_design),
        patch_suffix='',
        child_patch_run_one_args=tuple(child_patch_run_one_args),
        executed_feedback_text=f"{multiprocessing.current_process().name} Finished Initial Setup",
        do_run_model_TESTING=True,
        do_run_extraction_TESTING=True,
    )

# TEMP! This is for trialing new designs
new_design_trials: typing.List[typing.Tuple[generation.T_ProdNewGen, str]] = []

#for one_chain in chains.make_single_sided_chains(8):
#    one_forced_func = functools.partial(generation_FORCE.compel_new_generation, one_chain)
    # new_design_trials.append((one_forced_func, str(one_chain)))


new_design_trials.append((generation.produce_new_generation, "generation.produce_new_generation"))
# print(f"Doing {len(new_design_trials)} trails each time...")




def process_pool_run_and_process(run_one_args: generation.RunOneArgs) -> generation.RunOneArgs:
    """This returns a new version of the input argument, with info about the children filled in."""

    fn_odb = run_one_args.fn_inp.with_suffix('.odb')

    if run_one_args.do_run_model_TESTING: run_model(run_one_args.optim_params, run_one_args.fn_inp, force_single_core=False)

    fn_db_current = history.make_fn_in_dir(run_one_args.working_dir, ".db", run_one_args.iter_this, run_one_args.patch_suffix)

    if run_one_args.do_run_extraction_TESTING:
        # with mp_lock:
            perform_extraction(fn_odb, fn_db_current, run_one_args.nodal_z_override_in_odb, run_one_args.working_dir_extract, run_one_args.optim_params.add_initial_node_pos)

    # Sensitivity analysis with the "finite difference method"
    child_patch_run_one_args = []
    if run_one_args.is_full_model:

        patch_suffix_to_submod_infos = generation.produce_patch_models(run_one_args.working_dir, run_one_args.iter_this)
        all_run_one_args_in = []
        for patch_suffix, model_infos in patch_suffix_to_submod_infos.items():
            run_args_patch = run_one_args._replace(model_infos=model_infos, patch_suffix=patch_suffix)
            all_run_one_args_in.append(run_args_patch)

        # Recursive but only one layer deep! And fan out at this point to multi-core...
        if this_computer.n_abaqus_parallel_solves > 1:
            with multiprocessing.Pool(processes=this_computer.n_abaqus_parallel_solves, initializer=init, initargs=(mp_lock,)) as pool:
                for out_run_one_args in pool.imap_unordered(process_pool_run_and_process, all_run_one_args_in):
                    print(f"   " + out_run_one_args.executed_feedback_text)
                    child_patch_run_one_args.append(out_run_one_args)

        else:
            init(mp_lock)
            for run_args_patch in all_run_one_args_in:
                out_run_one_args = process_pool_run_and_process(run_args_patch)
                print(f"   " + out_run_one_args.executed_feedback_text)
                child_patch_run_one_args.append(out_run_one_args)

            # out_run_one_args = process_pool_run_and_process(run_args_patch, do_run_model=do_run_model, do_run_extraction=do_run_extraction)
            # child_patch_run_one_args.append(out_run_one_args)

    return run_one_args._replace(
        child_patch_run_one_args=tuple(child_patch_run_one_args),
        executed_feedback_text=f"{multiprocessing.current_process().name} Finished {run_one_args.fn_inp}"
    )


def init(mp_lock):
    global lock
    lock = mp_lock



def do_opt(stent_params: StentParams, optim_params: optimisation_parameters.OptimParams):
    working_dir = pathlib.Path(optim_params.working_dir)
    history_db_fn = history.make_history_db(working_dir)

    os.makedirs(working_dir, exist_ok=True)

    # If we've already started, use the most recent snapshot in the history.
    with history.History(history_db_fn) as hist:
        restart_i = hist.max_saved_iteration_num()
        start_from_scratch = restart_i is None

        if start_from_scratch:
            hist.set_stent_params(stent_params)
            hist.set_optim_params(optim_params)

        else:
            old_stent_params = hist.get_stent_params()
            if stent_params != old_stent_params:
                raise ValueError(r"Something changed with the stent params - can't use this one!")

    if start_from_scratch:
        run_one_args_completed = _from_scract_setup(working_dir, mp_lock)
        main_loop_start_i = 1

    else:
        print(f"Restarting from {restart_i}")
        main_loop_start_i = restart_i

    with history.History(history_db_fn) as hist:
        old_design = hist.get_most_recent_design()

    iter_prev = main_loop_start_i - 1
    previous_max_i = iter_prev

    while True:
        # Extract ONE from the previous generation
        one_design, model_info_to_rank = generation.process_completed_simulation(run_one_args_completed)
        if len(model_info_to_rank) != 1:
            raise ValueError("What?")

        one_ranking = next(iter(model_info_to_rank.values()))

        # for iter_this, (new_gen_func, new_gen_descip) in enumerate(new_design_trials, start=previous_max_i+1):
        iter_this = iter_prev + 1
        new_gen_descip = f"{optim_params.working_dir} {iter_this} {optim_params}"
        one_new_design = generation.produce_new_generation(working_dir, one_design, one_ranking, run_one_args_completed, new_gen_descip)

        new_elements = one_new_design.active_elements - one_design.active_elements
        removed_elements = one_design.active_elements - one_new_design.active_elements
        print(f"[{iter_prev} -> {iter_this}]\t{new_gen_descip}\tAdded: {len(new_elements)}\tRemoved: {len(removed_elements)}.")

        fn_inp = history.make_fn_in_dir(working_dir, ".inp", iter_this)

        construct_model.make_stent_model(optim_params, one_new_design, make_full_model_list(one_new_design), fn_inp)

        run_one_args_input = generation.RunOneArgs(
            working_dir=working_dir,
            optim_params=optim_params,
            iter_this=iter_this,
            nodal_z_override_in_odb=one_design.stent_params.nodal_z_override_in_odb,
            working_dir_extract=working_dir_extract,
            model_infos=[patch_manager.FullModelInfo(one_new_design)],
            patch_suffix='',
            child_patch_run_one_args=tuple(),
            executed_feedback_text='',
            do_run_model_TESTING=True,
            do_run_extraction_TESTING=True,
        )

        done = optim_params.is_converged(one_design, one_new_design, iter_this)

        MULTI_PROCESS_POOL = False
        if MULTI_PROCESS_POOL:
            with multiprocessing.Pool(processes=4, initializer=init, initargs=(mp_lock,)) as pool:

                for run_one_args_completed in pool.imap_unordered(process_pool_run_and_process, [run_one_args_input]):
                    print(run_one_args_completed.executed_feedback_text)

        else:
            init(mp_lock)
            run_one_args_completed = process_pool_run_and_process(run_one_args_input)
            print(run_one_args_completed.executed_feedback_text)

        iter_prev, one_design = iter_this, one_new_design

        if done:
            break

if __name__ == "__main__":

    stent_params = design.basic_stent_params
    optim_params = optimisation_parameters.active._replace(working_dir=str(this_computer.working_dir))
    # optim_params = optimisation_parameters.active._replace(working_dir=r"E:\Simulations\StentOpt\AA-256")

    # Check the serialisation
    optim_param_data = list(optim_params.to_db_strings())
    optim_param_data_again = optimisation_parameters.OptimParams.from_db_strings(optim_param_data)
    assert optim_params == optim_param_data_again

    stent_param_data = stent_params.json(indent=2)
    stent_param_again = design.StentParams.parse_raw(stent_param_data)
    assert stent_params == stent_param_again

    do_opt(stent_params, optim_params)

