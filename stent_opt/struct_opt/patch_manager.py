import abc
import collections
import dataclasses
import typing

from stent_opt.odb_interface import db_defs
from stent_opt.abaqus_model import amplitude

if typing.TYPE_CHECKING:
    from stent_opt.struct_opt import design

X, Y = 1, 2


T_DataList = typing.List[typing.Tuple[float, float]]

class PatchManager:
    """Handles the patch boundary conditions, etc"""
    node_dof_to_list: typing.Dict[typing.Tuple[int, int], T_DataList] = None
    nodes_we_have_results_for: typing.Set[int] = None
    node_num_to_pos: dict

    def __init__(self, node_num_to_pos: dict):
        self.node_dof_to_list = collections.defaultdict(list)
        self.nodes_we_have_results_for = set()
        self.node_num_to_pos = node_num_to_pos

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def ingest_node_pos(self, all_frames: typing.Iterable[db_defs.Frame], node_pos_rows: typing.Iterable[db_defs.NodePos]):

        # Look up the total simulation time from the frame
        frame_rowid_to_total_time = {frame.rowid: frame.simulation_time for frame in all_frames}
        node_pos_rows = list(node_pos_rows)  # For debugging
        for node_pos in node_pos_rows:
            self.nodes_we_have_results_for.add(node_pos.node_num)
            simulation_time = frame_rowid_to_total_time[node_pos.frame_rowid]
            node_initial_pos = self.node_num_to_pos[node_pos.node_num]
            for dof, val in ( (X, node_pos.X - node_initial_pos.x), (Y, node_pos.Y - node_initial_pos.y) ):
                key = (node_pos.node_num, dof)
                pair = (simulation_time, val)
                self.node_dof_to_list[key].append(pair)

    def _ordered_without_dupes(self, node_num: int, dof: int) -> T_DataList:

        # Have to be explicit about this because it's a default dict and will return an empty list
        key = (node_num, dof)
        if key not in self.node_dof_to_list:
            raise KeyError(key)

        working_list = self.node_dof_to_list[key]
        working_list.sort()

        # Squeeze out dupes, which can happen it seems.
        EPS = 1e-8
        def gen_points():
            prev_point = (-1234, -2345)  # I know I know but as if that's ever going to be in there...
            for point in working_list:
                time_decent_step = abs(point[0] - prev_point[0]) > EPS
                if time_decent_step:
                    prev_point = point
                    yield point

                else:
                    disp_close = abs(point[1] - prev_point[1]) < EPS
                    if disp_close:
                        # This is fine...
                        pass

                    else:
                        print(prev_point)
                        print(point)
                        raise ValueError(f"Got two points with close time but not close displacements... {prev_point} vs {point}.")

        return list(gen_points())

    def produce_amplitude_for(self, node_num: int, dof: int) -> amplitude.Amplitude:
        dof_name = {X: "U1", Y: "U2"}.get(dof)
        xy_data = tuple(amplitude.XY(x=pair[0], y=pair[1]) for pair in self._ordered_without_dupes(node_num, dof))
        return amplitude.Amplitude(
            name=f"N{node_num}-{dof_name}",
            data=xy_data,
            time_ref=amplitude.TimeReference.total_time,
        )

T_nodenum_dof_amp = typing.Tuple[int, typing.Dict[int, amplitude.Amplitude]]

@dataclasses.dataclass(unsafe_hash=True)
class SubModelInfoBase:
    stent_design: "design.StentDesign"
    boundary_node_nums: typing.FrozenSet[int]
    patch_manager: PatchManager
    node_elem_offset: int

    @abc.abstractmethod
    def elem_in_submodel(self, elem_num: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def node_in_submodel(self, node_num: int) -> bool:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_sub_model(self) -> bool:
        raise NotImplementedError()

    @property
    def is_full_model(self) -> bool:
        return not self.is_sub_model

    def boundary_node_enforced_displacements(self) -> typing.Iterable[T_nodenum_dof_amp]:
        # This will only ever do anything for the real submodel
        for node_num in self.boundary_node_nums:
            if node_num in self.patch_manager.nodes_we_have_results_for:
                this_node_dict = {dof: self.patch_manager.produce_amplitude_for(node_num, dof) for dof in (X, Y)}
                yield node_num, this_node_dict

    @abc.abstractmethod
    def real_to_model_node(self, node_num: int) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def real_to_model_elem(self, elem_num: int) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def model_to_real_node(self, node_num_submodel: int) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def model_to_real_elem(self, elem_num_submodel: int) -> int:
        raise NotImplementedError()

    def patch_elem_id_in_this_model(self, elem_num_submodel: int):
        real_elem_id = self.model_to_real_elem(elem_num_submodel)
        return self.elem_in_submodel(real_elem_id)

    def patch_node_id_in_this_model(self, node_num_submodel: int):
        real_node_id = self.model_to_real_node(node_num_submodel)
        return self.node_in_submodel(real_node_id)

    @abc.abstractmethod
    def all_patch_elem_ids_in_this_model(self) -> typing.Optional[typing.FrozenSet[int]]:
        raise NotImplementedError()


@dataclasses.dataclass(unsafe_hash=True)
class FullModelInfo(SubModelInfoBase):
    stent_design: "design.StentDesign"
    boundary_node_nums: typing.FrozenSet[int] = frozenset()
    patch_manager: PatchManager = None
    node_elem_offset: int = 0

    def elem_in_submodel(self, elem_num: int) -> bool:
        return True

    def node_in_submodel(self, node_num: int) -> bool:
        return True

    @property
    def is_sub_model(self) -> bool:
        return False

    def real_to_model_node(self, node_num: int) -> int:
        return node_num

    def real_to_model_elem(self, elem_num: int) -> int:
        return elem_num

    def model_to_real_node(self, node_num_submodel: int) -> int:
        return node_num_submodel

    def model_to_real_elem(self, elem_num_submodel: int) -> int:
        return elem_num_submodel

    def all_patch_elem_ids_in_this_model(self) -> typing.Optional[typing.FrozenSet[int]]:
        # Don't need to filter (just use all the elements in the model)
        return None


@dataclasses.dataclass(unsafe_hash=True)
class SubModelInfo(SubModelInfoBase):
    stent_design: "design.StentDesign"
    boundary_node_nums: typing.FrozenSet[int]
    patch_manager: PatchManager

    elem_nums: typing.FrozenSet[int]
    node_nums: typing.FrozenSet[int]
    reference_elem_num: int
    reference_elem_idx: "design.PolarIndex"
    initial_active_state: bool
    this_trial_active_state: bool
    node_elem_offset: int

    def __str__(self) -> str:
        return f"SubModelInfo(reference_elem_num={self.reference_elem_num}, node_elem_offset={self.node_elem_offset}, elem_nums={sorted(self.elem_nums)}"

    def elem_in_submodel(self, elem_num: int) -> bool:
        if elem_num == self.reference_elem_num:
            return self.this_trial_active_state

        return elem_num in self.elem_nums

    def node_in_submodel(self, node_num: int) -> bool:
        return node_num in self.node_nums

    @property
    def is_sub_model(self) -> bool:
        return True

    def make_inp_suffix(self) -> str:
        state = "On" if self.this_trial_active_state else "Off"
        return f"-N{self.reference_elem_num}-{state}"

    def real_to_model_node(self, node_num: int) -> int:
        return node_num + self.node_elem_offset

    def real_to_model_elem(self, elem_num: int) -> int:
        return elem_num + self.node_elem_offset

    def model_to_real_node(self, node_num_submodel: int) -> int:
        return node_num_submodel - self.node_elem_offset

    def model_to_real_elem(self, elem_num_submodel: int) -> int:
        return elem_num_submodel - self.node_elem_offset

    def all_patch_elem_ids_in_this_model(self) -> typing.Optional[typing.FrozenSet[int]]:
        # Do need to filter.
        working_elem_nums = {self.real_to_model_elem(elem_num_full_model) for elem_num_full_model in self.elem_nums}
        return frozenset(working_elem_nums)
