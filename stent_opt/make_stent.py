import multiprocessing
import os
import pathlib
import subprocess
import typing

import psutil

from stent_opt.struct_opt import design
from stent_opt.struct_opt import construct_model
from stent_opt.struct_opt.design import StentParams
from stent_opt.struct_opt import generation, optimisation_parameters
from stent_opt.struct_opt import patch_manager
# from stent_opt.struct_opt import generation_FORCE, chains

from stent_opt.struct_opt import history
from stent_opt.struct_opt.computer import this_computer


working_dir_orig = os.getcwd()
working_dir_extract = os.path.join(working_dir_orig, "odb_interface")

# This seems to be required to get
try:
    os.environ.pop('PYTHONIOENCODING')
except KeyError:
    pass

# TODO 2021-02-03
# - Graphs of volume ratios and target volume ratio
# - Rasterise and build (offline) the images of each step.


def kill_process_id(proc_id: int):
    process = psutil.Process(proc_id)
    for proc in process.children(recursive=True):
        proc.kill()

    process.kill()

def run_model(optim_params, inp_fn):
    old_working_dir = os.getcwd()


    path, fn = os.path.split(inp_fn)
    fn_solo = os.path.splitext(fn)[0]
    #print(multiprocessing.current_process().name, fn_solo)
    n_cpus = this_computer.n_cpus_abaqus_explicit if optim_params.is_explicit else this_computer.n_cpus_abaqus_implicit
    args = ['abaqus.bat', f'cpus={n_cpus}', f'job={fn_solo}']

    # This seems to need to come right after "job" in the argument list
    if optim_params.use_double_precision:
        args.append('double')

    args.extend(["ask_delete=OFF", 'interactive'])

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


def perform_extraction(odb_fn, out_db_fn, override_z_val, working_dir_extract):
    old_working_dir = os.getcwd()
    os.chdir(working_dir_extract)
    args = ["abaqus.bat", "cae", "noGui=odb_extract.py", "--", str(odb_fn), str(out_db_fn), str(override_z_val)]

    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ret_code = proc.wait()
    if ret_code:
        raise subprocess.SubprocessError(ret_code)



def _from_scract_setup(working_dir):
    history_db_fn = history.make_history_db(working_dir)

    # Do the initial setup from a first model.
    starting_i = 0
    fn_inp = history.make_fn_in_dir(working_dir, ".inp", starting_i)
    current_design = design.make_initial_design(stent_params)
    construct_model.make_stent_model(optim_params, current_design, patch_manager.FullModelInfo(), fn_inp)

    run_model(optim_params, fn_inp)
    perform_extraction(
        history.make_fn_in_dir(working_dir, ".odb", starting_i),
        history.make_fn_in_dir(working_dir, ".db", starting_i),
        current_design.stent_params.nodal_z_override_in_odb,
        working_dir_extract,
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


# TEMP! This is for trialing new designs
new_design_trials: typing.List[typing.Tuple[generation.T_ProdNewGen, str]] = []

#for one_chain in chains.make_single_sided_chains(8):
#    one_forced_func = functools.partial(generation_FORCE.compel_new_generation, one_chain)
    # new_design_trials.append((one_forced_func, str(one_chain)))


new_design_trials.append((generation.produce_new_generation, "generation.produce_new_generation"))
# print(f"Doing {len(new_design_trials)} trails each time...")

class RunOneArgs(typing.NamedTuple):
    working_dir: pathlib.Path
    optim_params: optimisation_parameters.OptimParams
    iter_this: int
    nodal_z_override_in_odb: float
    working_dir_extract: str


def process_pool_run_and_process(run_one_args: RunOneArgs):

    fn_inp = history.make_fn_in_dir(run_one_args.working_dir, ".inp", run_one_args.iter_this)
    fn_odb = history.make_fn_in_dir(run_one_args.working_dir, ".odb", run_one_args.iter_this)

    run_model(run_one_args.optim_params, fn_inp)

    fn_db_current = history.make_fn_in_dir(run_one_args.working_dir, ".db", run_one_args.iter_this)

    with lock:
        perform_extraction(fn_odb, fn_db_current, run_one_args.nodal_z_override_in_odb, run_one_args.working_dir_extract)

    generation.process_completed_simulation(run_one_args.working_dir, run_one_args.iter_this)

    return f"[{multiprocessing.current_process().name}] {run_one_args.iter_this} done."



def init(l):
    global lock
    lock = l



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
        _from_scract_setup(working_dir)
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
        one_design, one_ranking = generation.process_completed_simulation(working_dir, iter_prev)

        all_new_designs_this_iter = []
        done = False
        # Potentialy make many new models.

        arg_list = []
        for iter_this, (new_gen_func, new_gen_descip) in enumerate(new_design_trials, start=previous_max_i+1):
            one_new_design = new_gen_func(working_dir, one_design, one_ranking, iter_this, new_gen_descip)

            new_elements = one_new_design.active_elements - one_design.active_elements
            removed_elements = one_design.active_elements - one_new_design.active_elements
            print(f"[{iter_prev} -> {iter_this}]\t{new_gen_descip}\tAdded: {len(new_elements)}\tRemoved: {len(removed_elements)}.")

            fn_inp = history.make_fn_in_dir(working_dir, ".inp", iter_this)

            construct_model.make_stent_model(optim_params, one_new_design, patch_manager.FullModelInfo(), fn_inp)

            arg_list.append(RunOneArgs(working_dir, optim_params, iter_this, one_design.stent_params.nodal_z_override_in_odb, working_dir_extract))

            done = optim_params.is_converged(one_design, one_new_design, iter_this)

            all_new_designs_this_iter.append((iter_this, one_new_design))

            previous_max_i = max(previous_max_i, iter_this)

        MULTI_PROCESS_POOL = False
        l = multiprocessing.Lock()
        if MULTI_PROCESS_POOL:
            with multiprocessing.Pool(processes=4, initializer=init, initargs=(l,)) as pool:

                for res in pool.imap_unordered(process_pool_run_and_process, arg_list):
                    print(res)

            if len(all_new_designs_this_iter) > 1:
                warning = Warning("Arbitrary choice out of all new designs.")
                print(warning)

        else:
            init(l)
            for run_one_args in arg_list:
                process_pool_run_and_process(run_one_args)

        iter_prev, one_design = all_new_designs_this_iter[0]

        if done:
            break

if __name__ == "__main__":

    stent_params = design.basic_stent_params
    optim_params = optimisation_parameters.active._replace(working_dir=str(this_computer.working_dir))
    # optim_params = optimisation_parameters.active._replace(working_dir=r"E:\Simulations\StentOpt\AA-256")

    do_opt(stent_params, optim_params)

