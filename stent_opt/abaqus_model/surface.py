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
    SPOS = enum.auto()
    SNEG = enum.auto()


class Surface:
    name: str
    sets_and_faces: typing.List[ typing.Tuple[element.ElementSet, SurfaceFace]]

    def __init__(self, name: str, sets_and_faces: typing.List[ typing.Tuple[element.ElementSet, SurfaceFace]]):
        self.name = name
        self.sets_and_faces = sets_and_faces

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Surface, type=ELEMENT, name={self.name}"
        for elem_set, face in self.sets_and_faces:
            yield f"{elem_set.get_name(base.SetContext.assembly)}, {face.name}"

