import pathlib

from stent_opt.struct_opt import design, history
from stent_opt.struct_opt.generation import RankingResults




def produce_new_generation(working_dir: pathlib.Path, design_prev: design.StentDesign, ranking_result: RankingResults, iter_this: int, label: str) -> design.StentDesign:

    inp_fn = history.make_fn_in_dir(working_dir, ".inp", iter_this)
    history_db = history.make_history_db(working_dir)
    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()

    new_active_elems = evolve_decider(optim_params, design_prev, ranking_result.final_ranking_component, iter_this)

    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(stent_params.divs)}

    # Save the latest design snapshot
    # Log the history
    new_active_elem_nums = frozenset(elem_indices_to_num[idx] for idx in new_active_elems)
    with history.History(history_db) as hist:
        snapshot = history.Snapshot(
            iteration_num=iter_this,
            label=label,
            filename=str(inp_fn),
            active_elements=new_active_elem_nums)

        hist.add_snapshot(snapshot)

    return design.StentDesign(stent_params=design_prev.stent_params, active_elements=frozenset(new_active_elems), label=snapshot.label)
