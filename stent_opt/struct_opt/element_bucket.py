# Temporary (I think) to manually drive the iteration
import collections
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



def _build_graph(elems: typing.Iterable[Element]) -> networkx.Graph:

    """Confusingly, the "Nodes" in the graph are the elements in the FE mesh"""
    G = networkx.Graph()

    elems = list(elems)

    node_to_elems = collections.defaultdict(set)
    for elem in elems:
        for node in elem.conn:
            node_to_elems[node].add(elem)


    for fe_elem in elems:
        G.add_node(fe_elem)

    for fe_node, fe_elems in node_to_elems.items():
        for fe_elem1, fe_elem2 in itertools.permutations(fe_elems, 2):
            G.add_edge(fe_elem1, fe_elem2)

    return G


def network_traverse_test():
    elems = [
        Element(1, (1, 2, 3, 4)),
        Element(2, (4, 5, 6, 7)),
        Element(3, (10, 11, 30, 40)),
        Element(4, (3, 40, 67, 78)),
        Element(5, (1001, 1002, 1003, 1004)),
    ]

    G = _build_graph(elems)

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
    network_traverse_test()