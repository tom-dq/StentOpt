import itertools
import typing


class PolarIndex(typing.NamedTuple):
    R: int
    Th: int
    Z: int


class StentDesign(typing.NamedTuple):
    design_space: PolarIndex
    active_elements: typing.FrozenSet[PolarIndex]


def node_to_elem_design_space(divs: PolarIndex) -> PolarIndex:
    return divs._replace(
        R=divs.R - 1,
        Th=divs.Th - 1,
        Z=divs.Z - 1)


def node_from_index(divs: PolarIndex, iR, iTh, iZ):
    """Node numbers (fully populated)"""
    return (iR * (divs.Th * divs.Z) +
            iTh * divs.Z +
            iZ +
            1)  # One based node numbers!


def elem_from_index(divs: PolarIndex, iR, iTh, iZ):
    """Element numbers (fully populated) - just one fewer number in each ordinate."""
    divs_minus_one = node_to_elem_design_space(divs)
    return node_from_index(divs_minus_one, iR, iTh, iZ)


def generate_node_indices(divs: PolarIndex) -> typing.Iterable[typing.Tuple[int, PolarIndex]]:
    for iNode, (iR, iTh, iZ) in enumerate(itertools.product(
            range(divs.R),
            range(divs.Th),
            range(divs.Z)), start=1):

        yield iNode, PolarIndex(R=iR, Th=iTh, Z=iZ)


def generate_elem_indices(divs: PolarIndex) -> typing.Iterable[typing.Tuple[int, PolarIndex]]:
    divs_minus_one = node_to_elem_design_space(divs)
    yield from generate_node_indices(divs_minus_one)



def get_c3d8_connection(design_space: PolarIndex, i: PolarIndex):

    connection = [
        node_from_index(design_space, i.R, i.Th, i.Z),
        node_from_index(design_space, i.R, i.Th + 1, i.Z),
        node_from_index(design_space, i.R, i.Th + 1, i.Z + 1),
        node_from_index(design_space, i.R, i.Th, i.Z + 1),
        node_from_index(design_space, i.R + 1, i.Th, i.Z),
        node_from_index(design_space, i.R + 1, i.Th + 1, i.Z),
        node_from_index(design_space, i.R + 1, i.Th + 1, i.Z + 1),
        node_from_index(design_space, i.R + 1, i.Th, i.Z + 1),
    ]

    return tuple(connection)