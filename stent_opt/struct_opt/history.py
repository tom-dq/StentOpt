# Keeps track of how the optimisation has been going.
import itertools
import enum
import pathlib
import sqlite3
import typing
import statistics
import matplotlib.pyplot as plt

from stent_opt.struct_opt.common import RegionReducer
from stent_opt.abaqus_model import element
from stent_opt.struct_opt import design
from stent_opt.odb_interface import db_defs
from stent_opt.struct_opt import optimisation_parameters


_enum_types = [
    element.ElemType,
    RegionReducer
]

_nt_class_types = [
    db_defs.ElementStress,
    db_defs.ElementPEEQ,
]

class Snapshot(typing.NamedTuple):
    iteration_num: int
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
    primary = enum.auto()
    secondary = enum.auto()
    smoothed = enum.auto()


class StatusCheck(typing.NamedTuple):
    iteration_num: int
    elem_num: int
    stage: StatusCheckStage
    metric_name: str
    metric_val: float

    def for_db_form(self):
        return self._replace(stage=self.stage.name)

    def from_db(self):
        return self._replace(stage=StatusCheckStage[self.stage])


class DesignParam(typing.NamedTuple):
    param_name: str
    param_value: typing.Optional[str]


class OptParam(typing.NamedTuple):
    param_name: str
    param_value: typing.Optional[str]


class NodePosition(typing.NamedTuple):
    iteration_num: int
    node_num: int
    x: float
    y: float
    z: float


_make_snapshot_table = """CREATE TABLE IF NOT EXISTS Snapshot(
iteration_num INTEGER,
filename TEXT,
active_elements TEXT
)"""

_make_status_check_table = """CREATE TABLE IF NOT EXISTS StatusCheck(
iteration_num INTEGER,
elem_num INTEGER,
stage TEXT,
metric_name TEXT,
metric_val REAL
)"""

_make_design_parameters_table_ = """CREATE TABLE IF NOT EXISTS DesignParam(
param_name TEXT UNIQUE,
param_value TEXT)"""


_make_opt_parameters_table_ = """CREATE TABLE IF NOT EXISTS OptParam(
param_name TEXT UNIQUE,
param_value TEXT)"""

_make_node_pos_table_ = """CREATE TABLE IF NOT EXISTS NodePosition(
iteration_num INTEGER,
node_num INTEGER,
x REAL,
y REAL,
z REAL)"""

_make_tables = [
    _make_snapshot_table,
    _make_status_check_table,
    _make_design_parameters_table_,
    _make_opt_parameters_table_,
    _make_node_pos_table_,
]

class History:
    fn = None
    connection = None

    def __init__(self, fn):
        self.fn = fn
        self.connection = sqlite3.connect(fn)

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

    def get_snapshots(self) -> typing.Iterable[Snapshot]:
        with self.connection:
            rows = self.connection.execute("SELECT * FROM Snapshot ORDER BY iteration_num")
            db_forms = (Snapshot(*row) for row in rows)
            yield from (one_db_form.from_db() for one_db_form in db_forms)

    def set_stent_params(self, stent_params: "design.StentParams"):
        """Save the model parameters"""
        data_rows = list(stent_params.to_db_strings())
        ins_string = self._generate_insert_string_nt_class(DesignParam)
        with self.connection:
            self.connection.executemany(ins_string, data_rows)

    def get_stent_params(self) -> "design.StentParams":
        """Get the model parameters"""
        with self.connection:
            rows = self.connection.execute("SELECT * FROM DesignParam")
            l_rows = list(rows)

        return design.StentParams.from_db_strings(l_rows)

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
            )

    def get_status_checks(self, iter_greater_than: int) -> typing.Iterable[StatusCheck]:
        with self.connection:
            rows = self.connection.execute("SELECT * FROM StatusCheck WHERE iteration_num >= ? ORDER BY iteration_num, metric_name ", (iter_greater_than,))
            yield from (StatusCheck(*row).from_db() for row in rows)

    def get_node_positions(self) -> typing.Iterable[NodePosition]:
        with self.connection:
            rows = self.connection.execute("SELECT * FROM NodePosition ORDER BY iteration_num")
            yield from (NodePosition(*row) for row in rows)



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

def make_history_db(working_dir: pathlib.Path) -> pathlib.Path:
    return working_dir / "History.db"


def history_write_read_test():
    with History(r"c:\temp\aaa234.db") as history:
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
        for idx, one_item in enumerate(val):
            # Use the ellipsis to signal we don't know anything about the type of the element in the list - leave it up to the recursive call to deal with it.
            sub_key, sub_val = next(_item_to_db_strings(f"{key}.[{idx}]", one_item, ...))
            yield sub_key, sub_val

    elif len(matched_nt_class_types) == 1:
        # Just return the name to signal that it's the class.
        yield key, val.__name__

    else:
        raise TypeError(f"Don't known what to make of {key}: {val} type {this_item_type}.")


def nt_to_db_strings(nt_instance) -> typing.Iterable[typing.Tuple[str, typing.Optional[str]]]:
    """(key, value) pairs which can go into a database"""
    for key, val in nt_instance._asdict().items():
        this_item_type = _get_type_ignoring_nones(nt_instance._field_types[key])
        yield from _item_to_db_strings(key, val, this_item_type)


def single_item_from_string(s):
    """Strings which may be named tuples classes, or functions"""

    matches_nt_class = [nt for nt in _nt_class_types if nt.__name__ == s]
    matches_funcs = [f for f in optimisation_parameters._for_db_funcs if f.__name__ == s]
    matches = matches_nt_class + matches_funcs

    if len(matches) == 1:
        return matches[0]

    elif len(matches) > 1:
        raise ValueError(s)

    return None


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

    working_data = {}
    for prefix, data_sublist in itertools.groupby(sorted(data), key=get_prefix):
        if prefix:
            # Have to delegate to a child class to create.
            without_prefix_data = [remove_prefix(one_data) for one_data in data_sublist]

            nt_subclass = nt_class._field_types[prefix]
            single_type = _get_type_ignoring_nones(nt_subclass)

            # Is a sub-branch namedtuple or list of items.
            def is_an_item(k):
                return k.startswith('[') and k.endswith(']')

            text_is_a_list = all(is_an_item(k) for k,v in without_prefix_data)
            if text_is_a_list:
                working_data[prefix] = [single_item_from_string(nt_s) for k, nt_s in without_prefix_data]

            else:
                working_data[prefix] = single_type.from_db_strings(without_prefix_data)

        else:
            # Should be able to create it directly.
            for name, value in data_sublist:
                base_type = _get_type_ignoring_nones(nt_class._field_types[name])

                if value is None:
                    working_data[name] = None

                elif base_type in _enum_types:
                    # Is an enum, lookup from the dictionary.
                    working_data[name] = base_type[value]

                elif single_item_from_string(value):
                    working_data[name] = single_item_from_string(value)

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


if __name__ == "__main__":
    history_write_read_test()

if False:
    plot_history(r"E:\Simulations\StentOpt\aba-70\History.db")
