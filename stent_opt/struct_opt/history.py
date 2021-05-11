from __future__ import annotations

# Keeps track of how the optimisation has been going.
import itertools
import enum
import pathlib
import sqlite3
import typing
import statistics
import matplotlib.pyplot as plt

from stent_opt.struct_opt.common import RegionReducer, PrimaryRankingComponentFitnessFilter
from stent_opt.abaqus_model import element
from stent_opt.struct_opt import design
from stent_opt.odb_interface import db_defs
from stent_opt.struct_opt import optimisation_parameters

# TODO - fix up this stuff with enums not comparing equal: https://stackoverflow.com/a/40371520
_enum_types = [
    element.ElemType,
    RegionReducer,
    PrimaryRankingComponentFitnessFilter,
    optimisation_parameters.PostExpansionBehaviour,
]

_nt_class_types = [
    db_defs.ElementStress,
    db_defs.ElementPEEQ,
    db_defs.ElementEnergyElastic,
    db_defs.ElementEnergyPlastic,
    db_defs.ElementFatigueResult,
]


def _aggregate_elemental_values(st_vals: typing.Iterable[StatusCheck]) -> typing.Iterable[typing.Tuple[GlobalStatusType, float]]:
    vals = [stat_check.metric_val for stat_check in st_vals]

    for global_status_type in GlobalStatusType.get_elemental_aggregate_values():
        yield global_status_type, global_status_type.compute_aggregate(vals)


class Snapshot(typing.NamedTuple):
    iteration_num: int
    label: str
    filename: str
    active_elements: typing.FrozenSet[int]

    def for_db_form(self):
        return self._replace(
            active_elements=",".join(str(x) for x in self.active_elements)
        )

    def from_db(self):
        active_elements = frozenset(int(x) for x in self.active_elements.split(","))
        return self._replace(
            active_elements=active_elements,
        )

class StatusCheckStage(enum.Enum):
    filtering = enum.auto()
    primary = enum.auto()
    secondary = enum.auto()
    smoothed = enum.auto()


class StatusCheck(typing.NamedTuple):
    iteration_num: int
    elem_num: int
    stage: StatusCheckStage
    metric_name: str
    metric_val: float
    constraint_violation_priority: float

    def for_db_form(self):
        return self._replace(stage=self.stage.name)

    def from_db(self):
        return self._replace(stage=StatusCheckStage[self.stage])


class GlobalParam(typing.NamedTuple):
    param_name: str
    param_json: str


class OptParam(typing.NamedTuple):
    param_name: str
    param_value: typing.Optional[str]


class NodePosition(typing.NamedTuple):
    iteration_num: int
    node_num: int
    x: float
    y: float
    z: float


class GlobalStatusType(enum.Enum):
    abaqus_history_result = enum.auto()
    current_volume_ratio = enum.auto()
    target_volume_ratio = enum.auto()
    aggregate_min = enum.auto()
    aggregate_p10 = enum.auto()
    aggregate_mean = enum.auto()
    aggregate_median = enum.auto()
    aggregate_p90 = enum.auto()
    aggregate_max = enum.auto()
    aggregate_st_dev = enum.auto()
    aggregate_sum = enum.auto()

    @classmethod
    def get_elemental_aggregate_values(cls):
        for name, enum_obj in cls.__members__.items():
            if name.startswith("aggregate_"):
                yield enum_obj

    def compute_aggregate(self, elemental_vals: typing.List[float]) -> float:
        if self == GlobalStatusType.aggregate_min:
            return min(elemental_vals)

        elif self == GlobalStatusType.aggregate_max:
            return max(elemental_vals)

        elif self == GlobalStatusType.aggregate_mean:
            return statistics.mean(elemental_vals)

        elif self == GlobalStatusType.aggregate_median:
            return statistics.median(elemental_vals)

        elif self == GlobalStatusType.aggregate_st_dev:
            return statistics.stdev(elemental_vals)

        elif self == GlobalStatusType.aggregate_sum:
            return sum(elemental_vals)

        elif self in (GlobalStatusType.aggregate_p10, GlobalStatusType.aggregate_p90):
            quants = statistics.quantiles(elemental_vals, n=10)
            if self == GlobalStatusType.aggregate_p10:
                return quants[0]

            elif self == GlobalStatusType.aggregate_p90:
                return quants[-1]

        raise ValueError(f"Did not know what to return for {self}")


class PlottablePoint(typing.NamedTuple):
    iteration_num: int
    label: str
    value: float


class GlobalStatus(typing.NamedTuple):
    iteration_num: int
    global_status_type: GlobalStatusType
    global_status_sub_type: str
    global_status_value: float

    def for_db_form(self):
        return self._replace(global_status_type=self.global_status_type.name)

    def from_db(self):
        return self._replace(global_status_type=GlobalStatusType[self.global_status_type])

    def to_plottable_point(self) -> PlottablePoint:
        return PlottablePoint(iteration_num=self.iteration_num, label=f"{self.global_status_sub_type} {self.global_status_type.name}", value=self.global_status_value)

_make_snapshot_table = """CREATE TABLE IF NOT EXISTS Snapshot(
iteration_num INTEGER,
label TEXT,
filename TEXT,
active_elements TEXT
)"""

_make_status_check_table = """CREATE TABLE IF NOT EXISTS StatusCheck(
iteration_num INTEGER,
elem_num INTEGER,
stage TEXT,
metric_name TEXT,
metric_val REAL,
constraint_violation_priority REAL
)"""

_make_global_parameters_table_ = """CREATE TABLE IF NOT EXISTS GlobalParam(
param_name TEXT UNIQUE,
param_value JSON)"""


_make_opt_parameters_table_ = """CREATE TABLE IF NOT EXISTS OptParam(
param_name TEXT UNIQUE,
param_value TEXT)"""

_make_node_pos_table_ = """CREATE TABLE IF NOT EXISTS NodePosition(
iteration_num INTEGER,
node_num INTEGER,
x REAL,
y REAL,
z REAL)"""

_make_global_status_table = """CREATE TABLE IF NOT EXISTS GlobalStatus(
iteration_num INTEGER,
global_status_type TEXT,
global_status_sub_type TEXT,
global_status_value REAL
)"""

_make_tables = [
    _make_snapshot_table,
    _make_status_check_table,
    _make_global_parameters_table_,
    _make_opt_parameters_table_,
    _make_node_pos_table_,
    _make_global_status_table,
]

class History:
    fn = None
    connection = None

    def __init__(self, fn):
        self.fn = fn
        self.connection = sqlite3.connect(fn)
        self.connection.enable_load_extension(True)

        with self.connection:
            for mt in _make_tables:
                self.connection.execute(mt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.connection.close()


    def _generate_insert_string_nt_class(self, named_tuple_class):
        question_marks = ["?" for _ in named_tuple_class._fields]
        table_name = named_tuple_class.__name__
        return "INSERT INTO {0} VALUES ({1})".format(table_name, ", ".join(question_marks))

    def add_snapshot(self, snapshot: Snapshot):
        ins_string = self._generate_insert_string_nt_class(Snapshot)

        with self.connection:
            self.connection.execute(ins_string, snapshot.for_db_form())

    def add_many_status_checks(self, status_checks: typing.Iterable[StatusCheck]):
        ins_string = self._generate_insert_string_nt_class(StatusCheck)
        with self.connection:
            status_checks_for_db = (st.for_db_form() for st in status_checks)
            self.connection.executemany(ins_string, status_checks_for_db)

    def add_node_positions(self, node_positions: typing.Iterable[NodePosition]):
        ins_string = self._generate_insert_string_nt_class(NodePosition)
        with self.connection:
            self.connection.executemany(ins_string, node_positions)

    def add_many_global_status_checks(self, global_statuses: typing.Iterable[GlobalStatus]):
        global_statuses = list(global_statuses)
        with self.connection:
            ins_string = self._generate_insert_string_nt_class(GlobalStatus)

            with self.connection:
                in_db_forms = (glob_stat.for_db_form() for glob_stat in global_statuses)
                self.connection.executemany(ins_string, in_db_forms)

    def get_snapshots(self) -> typing.Iterable[Snapshot]:
        with self.connection:
            rows = self.connection.execute("SELECT * FROM Snapshot ORDER BY iteration_num, label")
            db_forms = (Snapshot(*row) for row in rows)
            yield from (one_db_form.from_db() for one_db_form in db_forms)

    def get_one_snapshot(self, iteration_num: int) -> Snapshot:
        with self.connection:
            rows = list(self.connection.execute("SELECT * FROM Snapshot WHERE iteration_num=? ORDER BY label", (iteration_num,)))
            if len(rows) != 1:
                raise ValueError(iteration_num)

            db_snap = Snapshot(*rows[0])
            return db_snap.from_db()

    def _set_global_param(self, some_param):
        param_name = some_param.__class__.__name__
        param_value = some_param.json(indent=2)

        ins_string = self._generate_insert_string_nt_class(GlobalParam)
        with self.connection:
            self.connection.execute(ins_string, (param_name,param_value,))

    def _get_global_param(self, some_param_class):
        with self.connection:
            rows = self.connection.execute("SELECT param_value FROM GlobalParam WHERE param_name = ?", (some_param_class.__name__,))
            l_rows = list(rows)
            assert len(l_rows) == 1
            data_json = l_rows[0][0]

        return some_param_class.parse_raw(data_json)

    def set_stent_params(self, stent_params: "design.StentParams"):
        """Save the model parameters"""
        self._set_global_param(stent_params)

    def get_stent_params(self) -> "design.StentParams":
        """Get the model parameters"""
        return self._get_global_param(design.StentParams)


    def set_optim_params(self, opt_params: optimisation_parameters.OptimParams):
        """Save the optimisation parameters."""
        data_rows = list(opt_params.to_db_strings())
        ins_string = self._generate_insert_string_nt_class(OptParam)
        with self.connection:
            self.connection.executemany(ins_string, data_rows)


    def get_opt_params(self) -> "optimisation_parameters.OptimParams":
        """Get the model parameters"""
        with self.connection:
            rows = self.connection.execute("SELECT * FROM OptParam")
            l_rows = list(rows)

        return optimisation_parameters.OptimParams.from_db_strings(l_rows)


    def max_saved_iteration_num(self) -> int:
        with self.connection:
            rows = self.connection.execute("SELECT iteration_num FROM Snapshot ORDER BY iteration_num DESC LIMIT 1")
            l_rows = list(rows)

        if len(l_rows) == 0:
            return None

        else:
            return l_rows[0][0]

    def _single_snapshot_row(self, rows) -> typing.Optional[Snapshot]:
        l_rows = list(rows)
        if len(l_rows) == 0:
            return None

        else:
            # There can be multiple snapshots because of restarts, etc. But they should be the same.
            final_form = {Snapshot(*row).from_db() for row in rows}

            if len(final_form) > 1:
                raise ValueError("More than one different Snapshot.")

            return final_form.pop()

    def get_snapshot(self, i: int) -> Snapshot:
        with self.connection:
            rows = list(self.connection.execute("SELECT * FROM Snapshot WHERE iteration_num = ?", (i,)))
            to_return = self._single_snapshot_row(rows)

        if not to_return:
            raise ValueError(f"Snapshot {i} not found.")

        return to_return

    def get_most_recent_snapshot(self) -> typing.Optional[Snapshot]:
        with self.connection:
            rows = list(self.connection.execute("SELECT * FROM Snapshot ORDER BY iteration_num DESC LIMIT 1"))
            return self._single_snapshot_row(rows)

    def get_most_recent_design(self) -> typing.Optional["design.StentDesign"]:
        maybe_snapshot = self.get_most_recent_snapshot()
        if maybe_snapshot:
            stent_params = self.get_stent_params()
            design_space = stent_params.divs
            elem_num_to_idx = {iElem: idx for iElem, idx in design.generate_elem_indices(design_space)}
            return design.StentDesign(
                stent_params=stent_params,
                active_elements=frozenset( (elem_num_to_idx[iElem] for iElem in maybe_snapshot.active_elements)),
                label=maybe_snapshot.label,
            )

    def get_status_checks(self, iter_greater_than_equal: int, iter_less_than_equal: int, limit_to_metrics: typing.Optional[typing.Container[str]]=None) -> typing.Iterable[StatusCheck]:
        with self.connection:
            if limit_to_metrics:
                qms = ["?" for _ in limit_to_metrics]
                qms_str = ", ".join(qms)
                extra_filter = f" AND metric_name IN ({qms_str}) "
                extra_args = tuple(limit_to_metrics)

            else:
                extra_filter = ""
                extra_args = tuple()

            all_args = (iter_greater_than_equal,iter_less_than_equal) + extra_args
            rows = self.connection.execute(f"SELECT * FROM StatusCheck WHERE iteration_num >= ? AND ? >= iteration_num {extra_filter} ORDER BY iteration_num, metric_name ", all_args)

            yield from (StatusCheck(*row).from_db() for row in rows)

    def get_global_status_checks(self, iter_greater_than_equal: int, iter_less_than_equal: int) -> typing.Iterable[GlobalStatus]:
        with self.connection:
            rows = self.connection.execute(f"SELECT * FROM GlobalStatus WHERE iteration_num >= ? AND ? >= iteration_num ORDER BY iteration_num, global_status_sub_type, global_status_type ", (iter_greater_than_equal, iter_less_than_equal))
            yield from (GlobalStatus(*row).from_db() for row in rows)

    def get_metric_names(self) -> typing.List[str]:
        with self.connection:
            rows = self.connection.execute("SELECT DISTINCT metric_name FROM StatusCheck")
            return list(row[0] for row in rows)

    def get_node_positions(self, iteration_num: typing.Optional[int] = None) -> typing.Iterable[NodePosition]:
        with self.connection:
            if iteration_num is None:
                rows = self.connection.execute("SELECT * FROM NodePosition ORDER BY iteration_num")

            else:
                rows = self.connection.execute("SELECT * FROM NodePosition WHERE iteration_num = ?", (iteration_num,))

            yield from (NodePosition(*row) for row in rows)


    def update_global_with_elemental(self, iteration_num: int):
        """Goes through the elemental results and gets the statistical values."""

        self.add_many_global_status_checks(self._make_global_status_rows_from_elemental(iteration_num))


    def _make_global_status_rows_from_elemental(self, iteration_num: int) -> typing.Iterable[GlobalStatus]:
        def stat_type(stat_check: StatusCheck):
            return stat_check.metric_name

        all_stat_checks = sorted(self.get_status_checks(iteration_num, iteration_num), key=stat_type)
        for metric_name, sub_list in itertools.groupby(all_stat_checks, stat_type):
            # Work out the statistics on sub_list

            for global_status_type, global_status_value in _aggregate_elemental_values(sub_list):
                yield GlobalStatus(
                    iteration_num=iteration_num,
                    global_status_type=global_status_type,
                    global_status_sub_type=metric_name,
                    global_status_value=global_status_value,
                )

    def get_unique_global_status_keys(self) -> typing.Iterable[GlobalStatus]:
        with self.connection:
            rows = self.connection.execute("SELECT DISTINCT global_status_type, global_status_sub_type FROM GlobalStatus")
            for global_status_type, global_status_sub_type in rows:
                yield GlobalStatus(
                    iteration_num=None,
                    global_status_type=GlobalStatusType[global_status_type],
                    global_status_sub_type=global_status_sub_type,
                    global_status_value=None,
                )

def plot_history(hist_fn):
    with History(hist_fn) as hist:
        all_status_lines = list(hist.get_status_checks())


    def metric_name_first(st):
        return st.metric_name

    def iteration_num_first(st):
        return st.iteration_num

    all_status_lines.sort(key=metric_name_first)

    # Hacky colour cycle
    all_colours = list(plt.get_cmap('tab10').colors)

    for metric_name, iter_of_metrics in itertools.groupby(all_status_lines, metric_name_first):
        one_metric_statuses_by_iter_num = sorted(iter_of_metrics, key=iteration_num_first)
        max_val = max(st.metric_val for st in one_metric_statuses_by_iter_num)

        x_axis = []
        y_max = []
        y_mean = []
        if max_val:
            normed_metrics = [st._replace(metric_val=st.metric_val/max_val) for st in one_metric_statuses_by_iter_num]

        else:
            normed_metrics = one_metric_statuses_by_iter_num

        for iter_num, it_of_metrics_by_elem in itertools.groupby(normed_metrics, iteration_num_first):
            all_vals = [st.metric_val for st in it_of_metrics_by_elem]

            x_axis.append(iter_num)
            y_max.append(max(all_vals))
            y_mean.append(statistics.mean(all_vals))

        this_col = all_colours.pop(0)


        plt.plot(x_axis, y_max, color=this_col, linestyle='--', label=f"{metric_name} Max")
        plt.plot(x_axis, y_mean, color=this_col, label=f"{metric_name} Mean")

    plt.gcf().set_size_inches(12, 9)
    plt.gca().set_yscale('log')

    plt.legend()
    plt.show()


def make_fn_in_dir(working_dir: pathlib.Path, ext: str, iter_num: int):
    """e.g., iter_num=123 and ext=".inp" """
    intermediary = working_dir / f'It-{str(iter_num).rjust(6, "0")}.XXX'
    return intermediary.with_suffix(ext)


def make_history_db(working_dir: typing.Union[str, pathlib.Path]) -> pathlib.Path:
    return pathlib.Path(working_dir) / "History.db"


def history_write_read_test():
    with History(r"c:\temp\aaabbbccc.db") as history:
        orig_stent_params = design.basic_stent_params
        history.set_stent_params(orig_stent_params)
        history.set_optim_params(optimisation_parameters.active)

        recreated_stent_params = history.get_stent_params()
        recreated_optim_params = history.get_opt_params()

        print(orig_stent_params == recreated_stent_params)
        print(optimisation_parameters.active == recreated_optim_params)


def _nt_has_all_fields_as_property(maybe_nt):
    fields = maybe_nt._fields

    def is_property(field_name):
        p = getattr(maybe_nt, field_name)
        print(p)

    if all(is_property(f_n) for f_n in fields):
        return True

    elif not any(is_property(f_n) for f_n in fields):
        return False

    else:
        raise ValueError("Indeterminate...")


def is_nt_class(maybe_nt) -> bool:
    try:
        return _nt_has_all_fields_as_property(maybe_nt) == True

    except AttributeError:
        # Not a named tuple
        return False


def is_nt_instance(maybe_nt) -> bool:
    try:
        return _nt_has_all_fields_as_property(maybe_nt) == False

    except AttributeError:
        # Not a named tuple
        return False


def _item_to_db_strings(key, val, this_item_type) -> typing.Iterable[typing.Tuple[str, typing.Optional[str]]]:
    matched_enum_types = [et for et in _enum_types if isinstance(val, et)]
    matched_nt_class_types = [nt for nt in _nt_class_types if nt is val]

    if this_item_type in (int, float, str):
        yield key, str(val)

    elif this_item_type == bool:
        yield key, val

    elif val is None:
        # Actual null in the DB.
        yield key, None

    elif hasattr(val, "to_db_strings"):
        # Delegate to the lower level
        for sub_key, sub_val in val.to_db_strings():
            yield f"{key}.{sub_key}", sub_val

    elif len(matched_enum_types) == 1:
        # Enums can also be stored - just use the name and we can figure out the type again later on.
        enum_type = matched_enum_types[0]
        yield key, val.name

    elif callable(val):
        # A function - assume it can be looked up or retrieved by the name.
        yield key, val.__name__

    elif isinstance(val, list):
        list_sub_type = this_item_type.__args__[0]
        for idx, one_item in enumerate(val):
            # Use the ellipsis to signal we don't know anything about the type of the element in the list - leave it up to the recursive call to deal with it.
            sub_key, sub_val = next(_item_to_db_strings(f"{key}.[{idx}]", one_item, list_sub_type))
            yield sub_key, sub_val

    elif len(matched_nt_class_types) == 1:
        # Just return the name to signal that it's the class.
        yield key, val.__name__

    else:
        raise TypeError(f"Don't known what to make of {key}: {val} type {this_item_type}.")


def nt_to_db_strings(nt_instance) -> typing.Iterable[typing.Tuple[str, typing.Optional[str]]]:
    """(key, value) pairs which can go into a database"""
    for key, val in nt_instance._asdict().items():
        this_item_type = _get_type_ignoring_nones(nt_instance.__annotations__[key])
        yield from _item_to_db_strings(key, val, this_item_type)


def single_item_from_string(s, maybe_target_type):
    """Strings which may be named tuples classes, or functions"""

    matches_nt_class = [nt for nt in _nt_class_types if nt.__name__ == s]
    matches_funcs = [f for f in optimisation_parameters._for_db_funcs if f.__name__ == s]

    is_an_enum = False
    try:
        is_an_enum = issubclass(maybe_target_type, enum.Enum)

    except TypeError:
        pass

    matches_enum = [e for e in maybe_target_type if e.name == s] if is_an_enum else []

    matches = matches_nt_class + matches_funcs + matches_enum

    if len(matches) == 1:
        return matches[0]

    elif len(matches) > 1:
        raise ValueError(s)

    return None


def _prepopulate_with_empty_lists(nt_class) -> dict:
    """If this is a named tuple subclass which has some potentially empty lists, need to put them in so it can be de-serialised without any problems from the DB."""

    EMPTY_ITERABLE_TYPES = [list, set, tuple]

    working_dict = {}

    for field_name, annotation in nt_class.__annotations__.items():
        maybe_origin = getattr(annotation, '__origin__', None)
        if maybe_origin in EMPTY_ITERABLE_TYPES:
            working_dict[field_name] = maybe_origin()

    return working_dict


def nt_from_db_strings(nt_class, data):

    def get_prefix(one_string_and_data: str):
        one_string, _ = one_string_and_data
        if "." in one_string:
            return one_string.split(".", maxsplit=1)[0]

        else:
            return None

    def remove_prefix(one_string_and_data: str):
        one_string, one_data = one_string_and_data
        if "." in one_string:
            return one_string.split(".", maxsplit=1)[1], one_data

        else:
            return None, one_data

    def get_class_or_none(base_type):
        """If base_type is typing.Type[AAA], returns AAA. Otherwise, None."""
        if getattr(base_type, "_name", None) != "Type":
            return None

        if not base_type.__args__:
            return None

        if len(base_type.__args__) != 1:
            raise ValueError("Not sure how to deal with this case!")

        return base_type.__args__[0]

    def is_typing_type(base_type):
        if get_class_or_none(base_type) is None:
            return False

        else:
            return True

    working_data = _prepopulate_with_empty_lists(nt_class)
    for prefix, data_sublist in itertools.groupby(sorted(data), key=get_prefix):
        if prefix:
            # Have to delegate to a child class to create.
            without_prefix_data = [remove_prefix(one_data) for one_data in data_sublist]

            nt_subclass = nt_class.__annotations__[prefix]
            single_type = _get_type_ignoring_nones(nt_subclass)

            # Is a sub-branch namedtuple or list of items.
            def is_an_item(k):
                return k.startswith('[') and k.endswith(']')

            text_is_a_list = all(is_an_item(k) for k,v in without_prefix_data)
            if text_is_a_list:
                working_data[prefix] = [single_item_from_string(nt_s, single_type.__args__[0]) for k, nt_s in without_prefix_data]

            else:
                working_data[prefix] = single_type.from_db_strings(without_prefix_data)

        else:
            # Should be able to create it directly.
            for name, value in data_sublist:
                base_type = _get_type_ignoring_nones(nt_class.__annotations__[name])

                if value is None:
                    working_data[name] = None

                elif _is_nullable(nt_class.__annotations__[name]) and value == "None":
                    # Get this for optional types
                    working_data[name] = None

                elif base_type in _enum_types:
                    # Is an enum, lookup from the dictionary.
                    working_data[name] = base_type[value]

                elif is_typing_type(base_type):
                    base_class = get_class_or_none(base_type)
                    # Could be any one of the subclasses...

                    match_subclasses = [sc for sc in base_class.__subclasses__() if sc.__name__ == value]
                    if not match_subclasses:
                        raise ValueError(f"Did not find anything called {value} among the subclasses of {base_class}.")

                    elif len(match_subclasses) == 1:
                        working_data[name] = match_subclasses[0]

                    else:
                        raise ValueError("What?")

                elif single_item_from_string(value, base_type):
                    working_data[name] = single_item_from_string(value, base_type)

                elif base_type == bool:
                    # Get a "0" string for false here...
                    working_data[name] = base_type(int(value))

                else:
                    working_data[name] = base_type(value)

    return nt_class(**working_data)


def _get_type_ignoring_nones(some_type):
    """Returns either a single type, or a tuple of types (special case for unions)."""
    if getattr(some_type, "__origin__", None) is typing.Union:
        non_none_args = [t for t in some_type.__args__ if t != type(None)]

    else:
        non_none_args = [some_type]

    if len(non_none_args) != 1:
        return tuple(non_none_args)
        # raise ValueError(some_type)

    return non_none_args[0]

def _is_nullable(some_type) -> bool:
    if getattr(some_type, "__origin__", None) is typing.Union:
        none_args = [t for t in some_type.__args__ if t == type(None)]
        if none_args:
            return True

    return False

if __name__ == "__main__":
    history_write_read_test()

    for x in GlobalStatusType.get_elemental_aggregate_values():
        print(x)

    #history_write_read_test()

if False:
    plot_history(r"E:\Simulations\StentOpt\aba-70\History.db")
