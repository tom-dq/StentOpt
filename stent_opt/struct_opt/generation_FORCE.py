import math
import pathlib
import typing

from stent_opt.struct_opt import design, history
from stent_opt.struct_opt import generation

from stent_opt.struct_opt import chains
from stent_opt.struct_opt import element_bucket


def make_sorted_chains(stent_params: design.StentParams, elem_idxs: typing.Iterable[design.PolarIndex]) -> typing.Dict[design.PolarIndex, float]:
    """given a bunch of elements, put them in "chains" and order them from 0 to 1."""

    cand_conns = {elem_idx: design.get_single_element_connection(stent_params, elem_idx) for elem_idx in elem_idxs}

    all_connected_elems_list = list(element_bucket.get_connected_elements(cand_conns))

    def minimum_many_elems(elem_idxs: typing.Iterable[design.PolarIndex]):
        return min(elem_idxs)

    # Sort them according to the minimum polar index of any element in the chain (hopefully this is relatively stable...)
    all_connected_elems_list.sort(key=minimum_many_elems)

    def produce_elems_in_order():
        for elem_set in all_connected_elems_list:
            yield from sorted(elem_set)

    overall_index_to_elem = {i: elem for i, elem in enumerate(produce_elems_in_order())}

    # Put it over the zero to one range
    delta = 1.0 / max(overall_index_to_elem)

    def make_out_val(i: int) -> float:
        if math.isclose(out_val := delta * i, 1.0):
            return 1.0

        else:
            return out_val

    out_dict = {elem: make_out_val(i) for i, elem in overall_index_to_elem.items()}
    return out_dict


def compel_new_generation(
        forced_changes: chains.Chains,
        working_dir: pathlib.Path,
        design_prev: design.StentDesign,
        ranking_result: generation.RankingResults,
        iter_this: int,
        label: str
) -> design.StentDesign:

    inp_fn = history.make_fn_in_dir(working_dir, ".inp", iter_this)
    history_db = history.make_history_db(working_dir)
    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()

    # Get the candidates for change.
    candidates_for = generation.get_candidate_elements(optim_params, design_prev)

    candidates = {
        chains.Action.add: candidates_for.introduction,
        chains.Action.remove: candidates_for.removal,
    }

    # Get the connectivity of these.
    action_to_changes = {}
    for action, cand_idxs in candidates.items():

        elem_to_x = make_sorted_chains(stent_params, cand_idxs)
        on_chain = {elem_idx for elem_idx, x in elem_to_x.items() if forced_changes.contains(action, x)}
        action_to_changes[action] = on_chain

    new_active_elems = (design_prev.active_elements | action_to_changes[chains.Action.add]) - action_to_changes[chains.Action.remove]

    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(stent_params.divs)}

    new_active_elem_nums = frozenset(elem_indices_to_num[idx] for idx in new_active_elems)
    with history.History(history_db) as hist:
        snapshot = history.Snapshot(
            iteration_num=iter_this,
            label=label,
            filename=str(inp_fn),
            active_elements=new_active_elem_nums)

        hist.add_snapshot(snapshot)

    return design.StentDesign(stent_params=design_prev.stent_params, active_elements=frozenset(new_active_elems), label=snapshot.label)
