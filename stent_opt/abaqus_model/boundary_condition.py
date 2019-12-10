import enum
import dataclasses
import typing

from stent_opt.abaqus_model import base, node


class BoundaryType(enum.Enum):
    XSYMM = enum.auto()
    YSYMM = enum.auto()
    ZSYMM = enum.auto()
    ENCASTRE = enum.auto()
    PINNED = enum.auto()


@dataclasses.dataclass(frozen=True)
class BoundaryBase:
    name: str

    def produce_inp_lines(self):
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class BoundarySymEncastre(BoundaryBase):
    name: str
    boundary_type: BoundaryType
    node_set: node.NodeSet

    def produce_inp_lines(self):
        set_name = self.node_set.get_name(base.SetContext.assembly)

        yield f"** Name: {self.name} Type: Symmetry/Antisymmetry/Encastre"
        yield "*Boundary"
        yield f"{set_name}, {self.boundary_type.name}"


class DispRotBoundComponent(typing.NamedTuple):
    node_set: node.NodeSet
    dof: int
    value: float


@dataclasses.dataclass(frozen=True)
class BoundaryDispRot(BoundaryBase):
    name: str
    components: typing.Tuple[DispRotBoundComponent, ...]

    def produce_inp_lines(self):
        yield f"** Name: {self.name} Type: Displacement/Rotation"
        yield "*Boundary"
        for comp in self.components:
            set_name = comp.node_set.get_name(base.SetContext.assembly)
            yield f"{set_name}, {comp.dof}, {comp.dof}, {base.abaqus_float(comp.value)}"
