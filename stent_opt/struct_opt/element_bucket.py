# Temporary (I think) to manually drive the iteration

import typing

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
    test_buckets()