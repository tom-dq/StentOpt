"""Simple graph for determining the connectivity between nodes, based on the elements they are connected to."""
import collections
import itertools
import enum

import networkx

from stent_opt.struct_opt import design


class _Entity(enum.Enum):
    node = enum.auto()
    elem = enum.auto()


def element_idx_to_nodes_within(distance: float, stent_design: "design.StentDesign"):
    """Gets elem -> set_of_nodes within some connectivity threshold."""
    return _element_idx_to_something_within(_Entity.node, distance, stent_design)


def element_idx_to_elems_within(distance: float, stent_design: "design.StentDesign"):
    """Gets elem -> set_of_elements within some connectivity threshold."""
    return _element_idx_to_something_within(_Entity.elem, distance, stent_design)


def _get_elem_idx_to_nodes(stent_design: "design.StentDesign"):
    elem_idx_to_nodes = {
        elem_idx: elem.connection
        for elem_idx, elem_num, elem in design.generate_stent_part_elements(
            stent_design.stent_params
        )
        if elem_idx in stent_design.active_elements
    }

    return elem_idx_to_nodes


def _get_elem_num_to_nodes(stent_design: "design.StentDesign"):
    elem_num_to_nodes = {
        elem_num: elem.connection
        for elem_idx, elem_num, elem in design.generate_stent_part_elements(
            stent_design.stent_params
        )
        if elem_idx in stent_design.active_elements
    }

    return elem_num_to_nodes


def _get_active_nodes(elem_idx_to_nodes):
    active_nodes = set()
    for node_list in elem_idx_to_nodes.values():
        active_nodes.update(node_list)

    return active_nodes


def _build_graph_from_design(stent_design: "design.StentDesign") -> networkx.Graph:
    elem_idx_to_nodes = _get_elem_idx_to_nodes(stent_design)
    active_nodes = _get_active_nodes(elem_idx_to_nodes)

    node_hops = dict()

    node_positions = design.get_node_num_to_pos(stent_design.stent_params)

    for node_list in elem_idx_to_nodes.values():

        # Always use the lowest node first for easy lookups
        for n1_n2 in itertools.combinations(sorted(node_list), 2):
            n1_n2 = tuple(n1_n2)
            positions = [node_positions[n_x].to_xyz() for n_x in n1_n2]
            xyz1, xyz2 = positions
            d_xyz = xyz1 - xyz2
            node_hops[n1_n2] = abs(d_xyz)

    graph = networkx.Graph()
    graph.add_nodes_from(active_nodes)
    for (n1, n2), d_xyz in node_hops.items():
        graph.add_edge(n1, n2, weight=d_xyz)

    # Add in the over-the-reflected-boundary nodes with zero weight.
    if stent_design.stent_params.wrap_around_theta:
        for n1, n2 in design.gen_active_pairs(stent_design.stent_params, active_nodes):
            graph.add_edge(n1, n2, weight=0.0)

    return graph


def _element_idx_to_something_within(
    entity: _Entity, distance: float, stent_design: "design.StentDesign"
):
    # TODO - do this a different way, with meshgrid and indices, for example.

    elem_num_to_idx = {
        iElem: idx
        for iElem, idx in design.generate_elem_indices(stent_design.stent_params.divs)
        if idx in stent_design.active_elements
    }

    # Get the hop distance between individual nodes.
    elem_idx_to_nodes = _get_elem_idx_to_nodes(stent_design)
    active_nodes = _get_active_nodes(elem_idx_to_nodes)

    graph = _build_graph_from_design(stent_design)

    # Get the node-to-nodes_within_radius

    OPT = 1

    node_to_nodes_within_radius = dict()
    if OPT == 0:
        for iNode in active_nodes:
            node_to_dist = networkx.single_source_dijkstra_path_length(
                graph, iNode, cutoff=distance
            )
            node_to_nodes_within_radius[iNode] = set(node_to_dist.keys())

    elif OPT == 1:
        for iSourceNode, iTargetNodes in networkx.all_pairs_dijkstra_path_length(
            graph, cutoff=distance
        ):
            node_to_nodes_within_radius[iSourceNode] = set(iTargetNodes)

    elif OPT == 2:
        # TODO - write this version!
        raise ValueError("Do this with meshgrid!")

    else:
        raise ValueError()

    # Get the nodes within the range.
    elem_idx_to_nodes_in_range = {}
    for elem_idx, node_connection in elem_idx_to_nodes.items():
        all_sets_of_nodes = [
            node_to_nodes_within_radius[iNode] for iNode in node_connection
        ]
        all_nodes_in_range = set.union(*all_sets_of_nodes)
        elem_idx_to_nodes_in_range[elem_idx] = all_nodes_in_range

    if entity == _Entity.node:
        return elem_idx_to_nodes_in_range

    elif entity == _Entity.elem:
        # Get the elements within the range.
        node_to_elem_idx = collections.defaultdict(set)
        for elem_idx, node_connection in elem_idx_to_nodes.items():
            for node in node_connection:
                node_to_elem_idx[node].add(elem_idx)

        elem_idx_to_elem_idx = collections.defaultdict(set)
        for elem_idx, all_nodes_in_range in elem_idx_to_nodes_in_range.items():
            all_elem_idxs_sets = [node_to_elem_idx[node] for node in all_nodes_in_range]
            elem_idx_to_elem_idx[elem_idx] = set.union(*all_elem_idxs_sets)

        return elem_idx_to_elem_idx

    else:
        raise ValueError(entity)


def get_elems_in_centre(stent_design: "design.StentDesign"):
    graph = _build_graph_from_design(stent_design)

    elem_idx_to_nodes = _get_elem_idx_to_nodes(stent_design)

    nodes_in_cent = set(networkx.center(graph))

    elems_in_cent = {
        elem
        for elem, nodes in elem_idx_to_nodes.items()
        if any(x in nodes_in_cent for x in nodes)
    }
    return elems_in_cent


def make_hole_away_from_boundary(stent_design: "design.StentDesign"):
    pass


if __name__ == "__main__":
    stent_params = design.basic_stent_params
    stent_design = design.make_initial_design(stent_params)

    aaa = get_elems_in_centre(stent_design)

    aaa = element_idx_to_nodes_within(0.2, stent_design)
    for k, v in aaa.items():
        print(k, v)
