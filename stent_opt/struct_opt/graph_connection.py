"""Simple graph for determining the connectivity between nodes, based on the elements they are connected to."""

import itertools

import networkx

from stent_opt.struct_opt import design

def element_nums_to_nodes_within(stent_params: design.StentParams, distance: float, stent_design: design.StentDesign):
    """Gets elem -> list_of_nodes within some connectivity threshold."""

    elem_num_to_idx = {iElem: idx for iElem, idx in design.generate_elem_indices(stent_design.stent_params.divs)}

    # Get the hop distance between individual nodes.
    elem_idx_to_nodes = {idx: design.get_c3d8_connection(stent_design.stent_params.divs, idx) for idx in elem_num_to_idx.values()}
    active_nodes = set()
    node_hops = dict()

    node_positions = design.generate_nodes_stent_polar(stent_params)

    for node_list in elem_idx_to_nodes.values():

        active_nodes.update(node_list)
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
    for n1, n2 in design.gen_active_pairs(stent_params, active_nodes):
        graph.add_edge(n1, n2, weight=0.0)

    # Get the node-to-nodes_within_radius
    node_to_nodes_within_radius = dict()
    for iNode in active_nodes:
        node_to_dist = networkx.single_source_dijkstra_path_length(graph, iNode, cutoff=distance)
        node_to_nodes_within_radius[iNode] = set(node_to_dist.keys())

    # Get the elements within the range.
    elem_idx_to_nodes_in_range = {}
    for elem_idx, node_connection in elem_idx_to_nodes.items():
        all_sets_of_nodes = [node_to_nodes_within_radius[iNode] for iNode in node_connection]
        all_nodes_in_range = set.union(*all_sets_of_nodes)
        elem_idx_to_nodes_in_range[elem_idx] = all_nodes_in_range

    return elem_idx_to_nodes_in_range


if __name__ == "__main__":
    stent_params = design.basic_stent_params
    stent_design = design.make_initial_design(stent_params)

    aaa = element_nums_to_nodes_within(stent_params, 0.2, stent_design)
    for k, v in aaa.items():
        print(k, v)



