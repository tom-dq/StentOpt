# Temporary (I think) to manually drive the iteration
import collections
import enum
import itertools
import typing

import networkx

class Element(typing.NamedTuple):
    num: int
    conn: typing.Tuple[int, ...]


class Bucket:
    _elems: typing.Set[Element]
    _all_nodes: typing.Set[int]

    def __init__(self, *elems: Element):
        self._elems = set(elems)
        self._full_node_update()

    def __contains__(self, elem):
        """The Bucket "Contains" the element if any node is shared with an existing element."""
        return any(node in self._all_nodes for node in elem.conn)

    def __add__(self, other):
        if not isinstance(other, Bucket):
            raise TypeError(other)

        # Empty buckets can be added no problem. But otherwise, check they overlap.
        should_check_nodes = bool(self._elems) and bool(other._elems)
        if should_check_nodes:
            shared_nodes = self._all_nodes & other._all_nodes
            if not shared_nodes:
                raise ValueError("Both buckets have elements in them but no nodes are shared.")

        # A bit hacky...
        working_bucket = Bucket()
        working_bucket._elems = self._elems | other._elems
        working_bucket._all_nodes = self._all_nodes | other._all_nodes

        return working_bucket

    def _full_node_update(self):
        self._all_nodes = {n for elem in self._elems for n in elem.conn}


class GraphEdgeEntity(enum.Enum):
    node = enum.auto()
    fe_elem_edge = enum.auto()


#                         [node,  fe_element_edge     ]
T_GraphEdge = typing.Union[int, typing.Tuple[int, int]]
def _generate_fe_elem_to_graph_edge(graph_edge_entity: GraphEdgeEntity, elems: typing.Iterable[Element]) -> typing.Iterable[
    typing.Tuple[Element, typing.List[T_GraphEdge]]]:

    if graph_edge_entity == GraphEdgeEntity.node:
        def edge_producer(elem: Element):
            for fe_node_num in elem.conn:
                yield fe_node_num

    elif graph_edge_entity == GraphEdgeEntity.fe_elem_edge:
        def edge_producer(elem: Element):
            node1 = elem.conn
            node2 = elem.conn[1:] + elem.conn[0:1]
            for n1, n2 in zip(node1, node2):
                yield tuple(sorted([n1, n2]))

    else:
        raise ValueError

    for elem in elems:
        yield elem, list(edge_producer(elem))



def build_graph(graph_edge_entity: GraphEdgeEntity, elems: typing.Iterable[Element]) -> networkx.Graph:

    """Confusingly, the "Nodes" in the graph are the elements in the FE mesh. And "graph_edges" is an interface between
    the finite elements. Could be a shared node or a shared edge."""
    G = networkx.MultiGraph()

    elems = list(elems)

    graph_edge_to_elems = collections.defaultdict(set)
    for elem, graph_edges in _generate_fe_elem_to_graph_edge(graph_edge_entity, elems):
        for graph_edge in graph_edges:
            graph_edge_to_elems[graph_edge].add(elem)

    for fe_elem in elems:
        G.add_node(fe_elem)

    for graph_edge, fe_elems in graph_edge_to_elems.items():
        for fe_elem1, fe_elem2 in itertools.permutations(fe_elems, 2):
            if graph_edge_entity.node:
                G.add_edge(fe_elem1, fe_elem2, fe_node_num=graph_edge)

            elif graph_edge_entity.fe_elem_edge:
                G.add_edge(fe_elem1, fe_elem2, fe_edge=graph_edge)

            else:
                raise ValueError(graph_edge_entity)

    return G



def get_fe_nodes_on_boundary_interface(g: networkx.Graph, k_cutoff: int, fe_elem_source: Element
            ) -> typing.Tuple[ typing.FrozenSet[int], typing.FrozenSet[Element] ]:

    def get_fe_elems_at_cutoff(k):
        all_elems_within = set()
        elems_on_interface = set()
        for fe_elem, level in networkx.single_source_shortest_path_length(g, fe_elem_source, k).items():
            all_elems_within.add(fe_elem)
            if k == level:
                elems_on_interface.add(fe_elem)

        return elems_on_interface, all_elems_within

    inside_interface, inside_all_fe_elems = get_fe_elems_at_cutoff(k_cutoff)
    outside_interface, _ = get_fe_elems_at_cutoff(k_cutoff+1)

    # Edges between the "inside" and "outside" elements are the interface nodes.
    boundary_nodes = set()
    for fe_elem_inside, fe_elem_outside, fe_node_num in networkx.edge_boundary(g, inside_interface, outside_interface, data="fe_node_num"):
        boundary_nodes.add(fe_node_num)

    return frozenset(boundary_nodes), frozenset(inside_all_fe_elems)



def test_boundary():
    elems = [
        Element(11, (111,1111,)),
        Element(22, (111,222,)),
        Element(33, (222,333,)),
        Element(44, (333,444,)),
        Element(55, (444,555,)),
        Element(66, (555,666,)),
        Element(77, (666,6666,)),
    ]

    g = build_graph(GraphEdgeEntity.node, elems)

    one_hop, _ = get_fe_nodes_on_boundary_interface(g, 1, elems[0])
    assert one_hop == frozenset({222,})


def test_bridge_gap():
    elems = [
        Element(11, (111, 1111,)),
        Element(22, (111, 222,)),
        Element(33, (222, 333,)),
        Element(44, (333, 444,)),
        Element(55, (444, 555,)),
        Element(66, (555, 666,)),
        Element(77, (666, 6666,)),
    ]

    g = build_graph(GraphEdgeEntity.node, elems)

    elem_trial = Element(88, (1111, 6666,))

    boundary_nodes, inside_all_fe_elems = get_fe_nodes_on_boundary_interface(g, 1, elem_trial)
    print(boundary_nodes)
    print(inside_all_fe_elems)

def network_traverse_test():
    elems = [
        Element(1, (1, 2, 3, 4)),
        Element(2, (4, 5, 6, 7)),
        Element(3, (10, 11, 30, 40)),
        Element(4, (3, 40, 67, 78)),
        Element(5, (1001, 1002, 1003, 1004)),
    ]

    G = build_graph(GraphEdgeEntity.node, elems)

    for c in networkx.connected_components(G):
        print(c)


    print(G)


TElem = typing.TypeVar("TElem")
def get_connected_elements(elem_to_conn: typing.Dict[TElem, typing.Tuple[int]]) -> typing.Iterable[ typing.Set[TElem] ]:
    G = networkx.Graph()

    node_to_elems = collections.defaultdict(set)
    for elem, nodes in elem_to_conn.items():
        for node in nodes:
            node_to_elems[node].add(elem)

    for fe_elem in elem_to_conn:
        G.add_node(fe_elem)

    for fe_node, fe_elems in node_to_elems.items():
        for fe_elem1, fe_elem2 in itertools.permutations(fe_elems, 2):
            G.add_edge(fe_elem1, fe_elem2)

    for c in networkx.connected_components(G):
        yield c


def test_buckets():
    e1a = Element(10, (1, 2, 3, 4))
    e1b = Element(11, (1, 2, 3, 5))

    e2 = Element(20, (10, 11, 12, 13))

    e_bridge = Element(15, (100, 1, 12, 23))

    b1 = Bucket(e1a, e1b)
    b2 = Bucket(e2)

    # Shouldn't be able to add these.
    got_error = False
    try:
        b1 + b2

    except ValueError:
        got_error = True

    assert got_error

    # Should be able to add them now.
    b_bridge = Bucket(e_bridge)
    b_int = b1 + b_bridge
    b_overall = b_int + b2

    assert len(b_overall._elems) == 4

    # Container checks
    assert e2 not in b1
    assert e_bridge in b1

    print("All OK")

if __name__ == "__main__":
    test_bridge_gap()