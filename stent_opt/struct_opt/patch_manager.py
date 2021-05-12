import abc
import collections
import typing

from stent_opt.odb_interface import db_defs
from stent_opt.abaqus_model import amplitude

X, Y = 0, 1


T_DataList = typing.List[typing.Tuple[float, float]]

class PatchManager:
    """Handles the patch boundary conditions, etc"""
    node_dof_to_list: typing.Dict[typing.Tuple[int, int], T_DataList] = None

    def __init__(self):
        self.node_dof_to_list = collections.defaultdict(list)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def ingest_node_pos(self, all_frames: typing.Iterable[db_defs.Frame], node_pos_rows: typing.Iterable[db_defs.NodePos]):

        # Look up the total simulation time from the frame
        frame_rowid_to_total_time = {frame.rowid: frame.simulation_time for frame in all_frames}
        for node_pos in node_pos_rows:
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
                time_close = abs(point[0] - prev_point[0]) > EPS
                if time_close:
                    prev_point = point
                    yield point

                else:
                    disp_close = abs(point[1] - prev_point[1]) > EPS
                    if disp_close:
                        # This is fine...
                        pass

                    else:
                        raise ValueError(f"Got two points with close time but not close displacements... {prev_point} vs {point}.")

        return list(gen_points())

    def produce_amplitude_for(self, node_num: int, dof: int) -> amplitude.Amplitude:
        dof_name = {X: "U1", Y: "U2"}.get(dof)
        xy_data = tuple(amplitude.XY(x=pair[0], y=pair[1]) for pair in self._ordered_without_dupes(node_num, dof))
        return amplitude.Amplitude(
            name=f"N{node_num}-{dof_name}",
            data=xy_data,
        )


class SubModelInfoBase:
    @abc.abstractmethod
    def elem_in_submodel(self, elem_num: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def node_enforced_displacement(self, node_num: int, dof: int) -> typing.Optional[amplitude.Amplitude]:
        raise NotImplementedError()


class FullModelInfo(SubModelInfoBase):
    def elem_in_submodel(self, elem_num: int) -> bool:
        return True

    def node_enforced_displacement(self, node_num: int, dof: int) -> typing.Optional[amplitude.Amplitude]:
        return None


class SubModelInfo(SubModelInfoBase):
    patch_manager: PatchManager = None
    elem_nums: typing.FrozenSet[int] = None
    boundary_node_nums: typing.FrozenSet[int] = None


    def __init__(self, patch_manager: PatchManager, elem_nums: typing.Iterable[int], boundary_nodes: typing.Iterable[int]):
        self.patch_manager = patch_manager
        self.elem_nums = frozenset(elem_nums)
        self.boundary_node_nums = frozenset(boundary_nodes)

    def elem_in_submodel(self, elem_num: int) -> bool:
        return elem_num in self.elem_nums

    def node_enforced_displacement(self, node_num: int, dof: int) -> typing.Optional[amplitude.Amplitude]:
        if node_num in self.boundary_node_nums:
            return self.patch_manager.produce_amplitude_for(node_num, dof)



