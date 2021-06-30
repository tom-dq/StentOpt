import sys

import solidspy

import memory_profiler

import numpy

from stent_opt.struct_opt import patch_manager
from stent_opt.struct_opt import design
from stent_opt.struct_opt import construct_model


# @memory_profiler.profile()
def patch_matrix_OK(sub_model_info: patch_manager.SubModelInfo):
    """checks if the FE mesh created by the submodel is going to be singular when we try to solve it..."""

    node_mapping = _make_node_mapping(sub_model_info)
    nodes = _build_nodes_array(sub_model_info, node_mapping)
    elements = _build_elems_array(sub_model_info, node_mapping)
    mats = numpy.array([[1.0, 0.3]])

    DME, IBC, neq = solidspy.assemutil.DME(nodes, elements)

    # System assembly
    KG = solidspy.assemutil.dense_assem(elements, mats, nodes, neq, DME)

    cond_num = numpy.linalg.cond(KG)

    MAX_CONDITION_NUM = 1/sys.float_info.epsilon
    VERY_BIG_COND_NUM = 0.001 * MAX_CONDITION_NUM

    is_ok = cond_num < VERY_BIG_COND_NUM

    is_ok_text = "ok" if is_ok else "SINGULAR"

    if not is_ok:
        print(f"{sub_model_info.reference_elem_num} {sub_model_info.this_trial_active_state} {is_ok_text} cond={cond_num}")

    # Why do I have to do this?
    del node_mapping
    del nodes
    del elements
    del mats
    del DME
    del IBC
    del neq
    del KG

    return is_ok


def _build_elems_array(sub_model_info: patch_manager.SubModelInfo, node_mapping):

    elems = []
    QUAD4 = 1
    MAT0 = 0
    for congi_idx, (idx, elem_num, one_elem) in enumerate(design.generate_stent_part_elements(sub_model_info.stent_design.stent_params)):
        if idx in sub_model_info.stent_design.active_elements and sub_model_info.elem_in_submodel(elem_num):
            elem_row = [congi_idx, QUAD4, MAT0] + [node_mapping[iNodeFull] for iNodeFull in one_elem.connection]
            elems.append(elem_row)

    return numpy.array(elems)


def _produce_nodes_with_gaps_in_numbers(sub_model_info: patch_manager.SubModelInfo):
    FREE, FIXED = 0, -1

    # Nodes are in the format [iNode, x, y x_fixed, y_fixed]
    stent_params = sub_model_info.stent_design.stent_params
    for iNodeFull, node_idx, xyz in design.generate_nodes(stent_params):
        is_within_design_domain = node_idx in sub_model_info.all_node_polar_index_admissible
        is_in_this_patch = iNodeFull in sub_model_info.node_nums
        if is_within_design_domain and is_in_this_patch:
            x_rest, y_rest = FREE, FREE
            # Restrained if on the patch boundary
            if iNodeFull in sub_model_info.boundary_node_nums:
                x_rest, y_rest = FIXED, FIXED

            # Restrained if ordained by the global model
            for global_node_set in construct_model.get_boundary_node_set_2d(
                    treat_spring_boundary_as_fixed=True,
                    submodel_boundary_nodes=sub_model_info.boundary_node_nums,
                    stent_params=sub_model_info.stent_design.stent_params,
                    reference_stent_design=sub_model_info.stent_design,
                    iNodeModel=iNodeFull,
                    node_idx=node_idx,
            ):
                if global_node_set.planar_x_is_constrained:
                    x_rest = FIXED

                if global_node_set.planar_y_is_constrained:
                    y_rest = FIXED

            yield iNodeFull, (xyz.x, xyz.y, x_rest, y_rest)


def _make_node_mapping(sub_model_info: patch_manager.SubModelInfo):
    return {
        iNodeFull: idx for idx, (iNodeFull, _) in enumerate(sorted(_produce_nodes_with_gaps_in_numbers(sub_model_info)))
    }


def _build_nodes_array(sub_model_info: patch_manager.SubModelInfo, node_mapping):
    nodes = []
    for iNodeFull, node_data in _produce_nodes_with_gaps_in_numbers(sub_model_info):
        iNodeContiguous = node_mapping[iNodeFull]
        node_row = [iNodeContiguous] + list(node_data)
        nodes.append(node_row)

    return numpy.array(nodes, dtype=float)

