import abc
import collections
import dataclasses
import typing

from stent_opt.odb_interface import db_defs
from stent_opt.abaqus_model import amplitude

X, Y = 1, 2


T_DataList = typing.List[typing.Tuple[float, float]]

class PatchManager:
    """Handles the patch boundary conditions, etc"""
    node_dof_to_list: typing.Dict[typing.Tuple[int, int], T_DataList] = None
    nodes_we_have_results_for: typing.Set[int] = None

    def __init__(self):
        self.node_dof_to_list = collections.defaultdict(list)
        self.nodes_we_have_results_for = set()

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
            for dof, val in ( (X, node_pos.X), (Y, node_pos.Y) ):
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

@dataclasses.dataclass()
class SubModelInfoBase:
    boundary_node_nums: typing.FrozenSet[int]
    patch_manager: PatchManager
    node_elem_offset: int

    @abc.abstractmethod
    def elem_in_submodel(self, elem_num: int) -> bool:
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




@dataclasses.dataclass()
class FullModelInfo(SubModelInfoBase):
    boundary_node_nums: typing.FrozenSet[int] = frozenset()
    patch_manager: PatchManager = None
    node_elem_offset: int = 0

    def elem_in_submodel(self, elem_num: int) -> bool:
        return True

    @property
    def is_sub_model(self) -> bool:
        return False

    def real_to_model_node(self, node_num: int) -> int:
        return node_num

    def real_to_model_elem(self, elem_num: int) -> int:
        return elem_num


@dataclasses.dataclass()
class SubModelInfo(SubModelInfoBase):
    boundary_node_nums: typing.FrozenSet[int]
    patch_manager: PatchManager
    stent_design: typing.Any

    elem_nums: typing.FrozenSet[int]
    reference_elem_num: int
    initial_active_state: bool
    this_trial_active_state: bool
    node_elem_offset: int


    def elem_in_submodel(self, elem_num: int) -> bool:
        if elem_num == self.reference_elem_num:
            return self.this_trial_active_state

        return elem_num in self.elem_nums

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

