import typing


from .base import XYZ, abaqus_float


class Nodes(dict):

    def __setitem__(self, key: int, value: XYZ):
        if not all(isinstance(x, float) for x in value):
            raise TypeError("Should only be floats in here.")

        super().__setitem__(key, value)

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield "*Node"
        for iNode, xyz in sorted(self.items()):
            yield f"{iNode:10}, {abaqus_float(xyz.x)}, {abaqus_float(xyz.y)}, {abaqus_float(xyz.z)}"
