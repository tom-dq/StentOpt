import enum
import typing

from stent_opt.abaqus_model import base, element

class SurfaceFace(enum.Enum):
    S1 = enum.auto()
    S2 = enum.auto()
    S3 = enum.auto()
    S4 = enum.auto()
    S5 = enum.auto()
    S6 = enum.auto()


class Surface:
    sets_and_faces: typing.List[ typing.Tuple[element.ElementSet, SurfaceFace]]

    def __init__(self):
        self.sets_and_faces = []

    def add_set(self, element_set: element.ElementSet, face: SurfaceFace):
        self.sets_and_faces.append( (element_set, face) )

    def generate_inp_lines(self) -> typing.Iterable[str]:
        pass