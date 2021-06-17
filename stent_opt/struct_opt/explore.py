from __future__ import annotations

import collections
import functools
import itertools
import operator
import pathlib
import typing
import enum

import holoviews
from holoviews import opts
import panel
import numpy

from matplotlib.figure import Figure
from matplotlib import rcParams

# TODO:
#  - Rasterize the output plots or something - make them snappier!
#  - Make an auto-animation?
#  - Something to do with stopping disconnections?

# TODO:
#   Plan for better investigation of what makes a good objective criteria:
#     - Refactor to make it possible to have a number of new generations from an old generation, with different parameters in each. Give new designs "labels" or something.
#     - See which criteria make the design better or worse
#     - Have multiple chains of elements being included or excluded.
#     - See which pushes the design in the right direction and see what local criteria achieved that.
#     - Nonlinear elastic as a proxy?


from stent_opt.abaqus_model import base, element
from stent_opt.struct_opt import history, design
from stent_opt.struct_opt.computer import this_computer

holoviews.extension('bokeh', 'matplotlib')


def _get_most_recent_working_dir() -> pathlib.Path:

    subdirs = (p for p in pathlib.Path(this_computer.base_working_dir).iterdir() if p.is_dir())

    def most_recent(p: pathlib.Path):
        return p.stat().st_ctime
        #text, num = p.name.split('-')[0:2]
        #return text, int(num), p.stat().st_ctime

    return max(subdirs, key=most_recent)


# WORKING_DIR_TEMP = _get_most_recent_working_dir()


# WORKING_DIR_TEMP = pathlib.Path(r"C:\Simulations\StentOpt\AA-43")
# WORKING_DIR_TEMP = pathlib.Path(r"C:\Simulations\StentOpt\AA-46")
# WORKING_DIR_TEMP = pathlib.Path(r"C:\Simulations\StentOpt\AA-48")
# WORKING_DIR_TEMP = pathlib.Path(r"C:\Simulations\StentOpt\AA-49")
# WORKING_DIR_TEMP = pathlib.Path(r"C:\Simulations\StentOpt\AA-51")
# WORKING_DIR_TEMP = pathlib.Path(r"E:\Simulations\StentOpt\AA-125")

# WORKING_DIR_TEMP = pathlib.Path(r"C:\Simulations\StentOpt\AA-180")
WORKING_DIR_TEMP = pathlib.Path(r"E:\Simulations\StentOpt\AA-285")

# WORKING_DIR_TEMP = pathlib.Path(r"C:\Simulations\StentOptDesktop\AA-229")


UNLIMITED = 1_000_000_000_000  # Should be enough
STOP_AT_INCREMENT = 1000


# Just re-use this around the place so I don't need to open/close the DB all the time... is this a bad idea?
history_db = history.make_history_db(WORKING_DIR_TEMP)
global_hist = history.History(history_db)

def _update_global_db_data():
    global _all_global_statuses, _global_status_idx, _all_elemental_metrics, _max_dashboard_increment
    _all_global_statuses = [gsv.to_plottable_point() for gsv in global_hist.get_unique_global_status_keys()]
    _global_status_idx = {pp.label: idx for idx, pp in enumerate(_all_global_statuses)}
    _all_elemental_metrics = global_hist.get_metric_names()
    _max_dashboard_increment = min(STOP_AT_INCREMENT, global_hist.max_saved_iteration_num() - 1)  # Why the minus one? Can't remember!
    if _max_dashboard_increment == 0:
        _max_dashboard_increment = 1  # Special case to make the slider work if there aren't enough results


_update_global_db_data()


class DeformationView(enum.Enum):
    undeformed_active = "Undeformed (Active only)"
    undeformed_all = "Undeformed (All with results)"
    deformed_active = "Deformed (Active only)"

    @property
    def should_show_ghosts(self) -> bool:
        return self == DeformationView.undeformed_all

    @property
    def is_deformed(self) -> bool:
        return self == DeformationView.deformed_active

class ContourView(typing.NamedTuple):
    iteration_num: int
    metric_name: str
    deformation_view: DeformationView
    min_val: typing.Optional[float]
    max_val: typing.Optional[float]

    def make_iteration_view(self) -> ContourIterationView:
        return ContourIterationView(metric_name=self.metric_name, deformation_view=self.deformation_view)

    def get_cnorm(self) -> str:
        return 'linear'
        if self.min_val <= 0:
            return 'linear'
        else:
            return 'log'


class ContourIterationView(typing.NamedTuple):
    metric_name: str
    deformation_view: DeformationView


class GraphLine(typing.NamedTuple):
    points: typing.List[history.PlottablePoint]

    @property
    def label(self):
        labels = {p.label for p in self.points}
        if len(labels) == 0:
            return None

        elif len(labels) == 1:
            return labels.pop()

        else:
            raise ValueError(labels)

    @property
    def x_vals(self):
        return [p.iteration_num for p in self.points]

    @property
    def y_vals(self):
        return [p.value for p in self.points]

    @property
    def xy_points(self):
        return [(p.iteration_num, p.value) for p in self.points]

    @property
    def y_range(self):
        return min(self.y_vals), max(self.y_vals)


class GraphLineCollection(typing.NamedTuple):
    graph_lines: typing.List[GraphLine]
    y_min: typing.Union[float, Ellipsis]
    y_max: typing.Union[float, Ellipsis]

    @staticmethod
    def no_ellipses(x1, x2):
        maybes = [x1, x2]
        reals = [x for x in maybes if x != Ellipsis]
        return reals

    def __add__(self, other: GraphLineCollection) -> GraphLineCollection:
        return GraphLineCollection(
            graph_lines=self.graph_lines + other.graph_lines,
            y_min=max(self.no_ellipses(self.y_min, other.y_min), default=...),
            y_max=min(self.no_ellipses(self.y_max, other.y_max), default=...),
        )

    @staticmethod
    def get_identity() -> GraphLineCollection:
        return GraphLineCollection(
            graph_lines=[],
            y_min=...,
            y_max=...,
        )

    @staticmethod
    def from_single(graph_line: GraphLine) -> GraphLineCollection:
        return GraphLineCollection(
            graph_lines=[graph_line],
            y_min=min(graph_line.y_vals),
            y_max=max(graph_line.y_vals),
        )

    @staticmethod
    def from_many_graphs(graph_lines: typing.Iterable[GraphLine]) -> GraphLineCollection:
        single_graph_collections = [GraphLineCollection.from_single(gl) for gl in graph_lines]

        return functools.reduce(operator.add, single_graph_collections, GraphLineCollection.get_identity())

    @property
    def y_range(self) -> typing.Tuple[float, float]:
        if self.y_min == Ellipsis or self.y_max == Ellipsis:
            raise ValueError(self)

        return self.y_min, self.y_max

def get_status_checks() -> typing.List["history.StatusCheck"]:
    return list(global_hist.get_status_checks(iter_greater_than_equal=0, iter_less_than_equal=UNLIMITED))


def _build_contour_view_data(
        hist: history.History,
        single_iteration: typing.Optional[int] = None,
        metric_name: typing.Optional[str] = None,
) -> typing.Iterable[typing.Tuple[ContourView, typing.Dict[int, float]]]:
    """Make the view info for a contour."""

    def make_contour_view(status_check: history.StatusCheck) -> ContourView:
        return ContourView(
            iteration_num=status_check.iteration_num,
            metric_name=status_check.metric_name,
            deformation_view=None,
            min_val=None,
            max_val=None,
        )

    if metric_name:
        good_metric_names = [metric_name]

    else:
        metric_names = hist.get_metric_names()
        # good_metric_names = ["ElementEnergyElastic"]
        good_metric_names = metric_names

    if single_iteration is None:
        inc_gt = 0
        inc_lte = STOP_AT_INCREMENT

    else:
        inc_gt = single_iteration
        inc_lte = single_iteration

    for contour_view, sub_iter in itertools.groupby(hist.get_status_checks(inc_gt, inc_lte, good_metric_names), make_contour_view):
        min_all_frames, max_all_frames = hist.get_min_max_status_check(contour_view.metric_name)
        contour_view_with_limits = contour_view._replace(min_val=min_all_frames, max_val=max_all_frames)
        elem_vals = {status_check.elem_num: status_check.metric_val for status_check in sub_iter}
        yield contour_view_with_limits, elem_vals


def make_quadmesh(
        stent_params: design.StentParams,
        active_elements: typing.FrozenSet[design.PolarIndex],
        node_idx_pos: typing.Dict[design.PolarIndex, base.XYZ],
        contour_view: ContourView,
        elem_idx_to_value: typing.Dict[design.PolarIndex, float],
        title: str,
) -> holoviews.Overlay:
    """Make a single Quadmesh representation"""

    # Make the undeformed meshgrid.
    x_vals = design.gen_ordinates(stent_params.divs.Th, 0.0, stent_params.theta_arc_initial)
    y_vals = design.gen_ordinates(stent_params.divs.Z, 0.0, stent_params.length)

    # Get the bounds which are non-NaN
    idx_z = [idx.Z for idx in elem_idx_to_value.keys()]
    idx_Th = [idx.Th for idx in elem_idx_to_value.keys()]

    z_low, z_high = min(idx_z), max(idx_z)+1  # This plus one is for element to upper node index
    th_low, th_high = min(idx_Th), max(idx_Th)+1

    x_vals_trim = x_vals[th_low: th_high+1]  # This plus one is for Python indexing.
    y_vals_trim = y_vals[z_low: z_high+1]

    # X and Y are boundary points.
    X, Y = numpy.meshgrid(x_vals_trim, y_vals_trim)

    # The values are element centred.
    elem_shape = (X.shape[0]-1, X.shape[1]-1)
    Z = numpy.full(shape=elem_shape, fill_value=numpy.nan)
    Z_ghost = Z.copy()

    for idx, val in elem_idx_to_value.items():
        if idx in active_elements:
            Z[idx.Z-z_low, idx.Th-th_low] = val

        elif contour_view.deformation_view.should_show_ghosts:
            Z_ghost[idx.Z - z_low, idx.Th - th_low] = val

    all_ghosts_nan = numpy.isnan(Z_ghost).all()

    # Overwrite with the nodes...
    th_range = range(th_low, th_high+1)
    z_range = range(z_low, z_high+1)
    for node_idx, pos in node_idx_pos.items():
        if node_idx.Th in th_range and node_idx.Z in z_range:
            X[node_idx.Z-z_low, node_idx.Th-th_low] = pos.x
            Y[node_idx.Z-z_low, node_idx.Th-th_low] = pos.y

    # Populate with the real values.

    qmesh_real = holoviews.QuadMesh((X, Y, Z), vdims='level', group=contour_view.metric_name).redim.range(level=(contour_view.min_val, contour_view.max_val))
    qmesh_real.options(cmap='viridis')

    qmesh_list = [qmesh_real]
    if contour_view.deformation_view.should_show_ghosts and not all_ghosts_nan:
        qmesh_ghost = holoviews.QuadMesh((X, Y, Z_ghost), vdims='level', group=contour_view.metric_name)
        qmesh_ghost.options(cmap='inferno')
        qmesh_list.append(qmesh_ghost)

    for qmesh in qmesh_list:
        qmesh.opts(
            aspect='equal',
            line_width=0.1,
            padding=0.1,
            width=this_computer.fig_size[0],
            height=this_computer.fig_size[1],
            colorbar=True,
            bgcolor='lightgray',
            title=title,
            cnorm=contour_view.get_cnorm(),
        )

    return holoviews.Overlay(qmesh_list)

# Controls out here since iteration_selector is shared
iteration_selector = panel.widgets.IntSlider(
    name="Iteration Number",
    start=0,
    end=_max_dashboard_increment,
    step=1,
    value=0,
)

elemental_metric_selector = panel.widgets.Select(
    name="Contour",
    options=_all_elemental_metrics,
    value=_all_elemental_metrics[0],
    size=len(_all_elemental_metrics),
)

deformation_view_selector = panel.widgets.RadioButtonGroup(
    name="Deformation View",
    options=[dv.value for dv in DeformationView],
    value=list(DeformationView)[0].value
)

history_plot_vars_selector = panel.widgets.MultiSelect(
    name="History Values",
    options=[gsv.label for gsv in _all_global_statuses],
    size=50,
    value=[gsv.label for gsv in _all_global_statuses[0:1]],
    width=500,
)

def _update_controls():
    _update_global_db_data()
    iteration_selector.end = _max_dashboard_increment

@panel.depends(iteration_selector, elemental_metric_selector, deformation_view_selector)
def _make_single_contour(*args, **kwargs) -> holoviews.Overlay:
    _update_controls()

    stent_params = global_hist.get_stent_params()

    # Parameters
    iteration_num = iteration_selector.value
    deformation_view_value = deformation_view_selector.value
    metric_name = elemental_metric_selector.value

    deformation_view = DeformationView(deformation_view_value)

    one_contour = _contour_build_from_db(stent_params, iteration_num, deformation_view, metric_name)
    print(iteration_num, deformation_view, metric_name)
    return one_contour

def _make_animation():
    stent_params = global_hist.get_stent_params()

    # Parameters
    deformation_view_value = deformation_view_selector.value
    metric_name = elemental_metric_selector.value

    # deformation_view = DeformationView(deformation_view_value)
    deformation_view = DeformationView.deformed_active

    for iteration_num in range(_max_dashboard_increment):
        print(iteration_num)
        one_contour = _contour_build_from_db(stent_params, iteration_num, deformation_view, metric_name)
        iter_str = str(iteration_num).rjust(3, "0")
        holoviews.save(one_contour, fr"C:\Temp\aaaholo\{WORKING_DIR_TEMP.parts[-1]}-{metric_name}-{deformation_view.name}-{iter_str}.png")

@functools.lru_cache(1024)
def _contour_build_from_db(stent_params: design.StentParams, iteration_num: int, deformation_view: DeformationView, metric_name: str) -> holoviews.Overlay:
    node_num_to_idx = {iNode: idx for iNode, idx, pos in design.generate_nodes(stent_params)}
    elem_num_to_idx = {iElem: idx for idx, iElem, elem in design.generate_stent_part_elements(stent_params)}

    snapshot = global_hist.get_one_snapshot(iteration_num)

    # Node positions
    if deformation_view.is_deformed:
        node_pos = {
            node_pos.node_num: base.XYZ(x=node_pos.x, y=node_pos.y, z=node_pos.z)
            for node_pos in global_hist.get_node_positions(iteration_num)
        }

    else:
        node_pos = {iNode: pos.to_xyz() for iNode, idx, pos in design.generate_nodes(stent_params)}


    if deformation_view.should_show_ghosts:
        active_elements = frozenset(elem_num_to_idx.values())

    else:
        active_elements = frozenset(elem_num_to_idx[iElem] for iElem in snapshot.active_elements)

    # Should just be one contour here
    one_contour_view = list(_build_contour_view_data(global_hist, single_iteration=iteration_num, metric_name=metric_name))
    if len(one_contour_view) != 1:
        raise ValueError(f"Expected one here.")

    contour_view, elem_data = one_contour_view[0]
    contour_view = contour_view._replace(deformation_view=deformation_view)

    # iElem in elem_num_to_idx is for the smoothed results with sym planes, etc.
    elem_idx_to_value = {elem_num_to_idx[iElem]: val for iElem, val in elem_data.items() if iElem in elem_num_to_idx and elem_num_to_idx[iElem] in active_elements}

    node_idx_pos = {node_num_to_idx[iNode]: pos for iNode, pos in node_pos.items()}
    qmesh = make_quadmesh(stent_params, active_elements, node_idx_pos, contour_view, elem_idx_to_value, snapshot.label)

    return qmesh


@panel.depends(iteration_selector, history_plot_vars_selector)
def _create_history_curve(*args, **kwargs) -> holoviews.NdOverlay:

    _update_controls()

    graph_line_collection = _create_graph_line_collection()

    iteration_num = iteration_selector.value
    iteration_num_line = holoviews.VLine(iteration_num).opts(color='grey', line_dash='dashed')

    curves = {}
    print(f"{len(graph_line_collection.graph_lines)} lines on graph:")
    for graph_line in graph_line_collection.graph_lines:
        print("   ", graph_line.label, f"{graph_line.y_range}", graph_line.xy_points[0:4])
        curves[graph_line.label] = holoviews.Curve(graph_line.xy_points, 'iteration_num', graph_line.label, label=graph_line.label)

    # nd_overlay = holoviews.NdOverlay(curves, kdims=['metric'])
    # For some reason NDOverlap is flaky here - do it another way...
    overlay = functools.reduce(operator.mul, curves.values())

    with_opts = (
        overlay
            .opts(opts.Curve(framewise=True))  # Makes the curves re-adjust themselves with each update.
            .opts(
                # legend_position='right',
                width=this_computer.fig_size[0],
                height=int(0.5 * this_computer.fig_size[1]),
                padding=0.1,
                framewise=True,
            )
    )

    print(with_opts)

    print()

    return with_opts * iteration_num_line


def _create_graph_line_collection() -> GraphLineCollection:

    # Widgets
    history_plot_vars = set(history_plot_vars_selector.value)

    print(history_plot_vars)

    all_gsvs = [gsv for gsv in global_hist.get_global_status_checks(0, UNLIMITED)]
    all_pps = [gsv.to_plottable_point() for gsv in all_gsvs]

    plot_points = [pp for pp in all_pps if pp.label in history_plot_vars]
    graph_points = collections.defaultdict(list)

    for pp in plot_points:
        graph_points[pp.label].append(pp)

    # If there's only one point in there, add another one so it comes up on the graph.
    for points in graph_points.values():
        if len(points) == 1:
            pp_single = points[0]
            pp0 = pp_single._replace(iteration_num=pp_single.iteration_num-1)
            points.insert(0, pp0)

    graph_lines = [GraphLine(points) for points in graph_points.values()]

    return GraphLineCollection.from_many_graphs(graph_lines)


def _plot_view_func(graph_lines: typing.List[GraphLine], iteration_num: int):

    fig_dpi = rcParams['figure.dpi']

    fig_inches = this_computer.fig_size[0]/fig_dpi, 0.4*this_computer.fig_size[1]/fig_dpi

    fig = Figure(figsize=fig_inches, dpi=fig_dpi)
    ax = fig.add_subplot()

    for graph_line in graph_lines:
        x_vals = graph_line.x_vals
        y_vals = graph_line.y_vals
        ax.plot(x_vals, y_vals, label=graph_line.label)
        ax.plot([x_vals[iteration_num]], [y_vals[iteration_num]], 'o')

    return fig



def make_dashboard(working_dir: pathlib.Path):

    print(working_dir)

    dmap = holoviews.DynamicMap(_make_single_contour)
    hist_map = holoviews.DynamicMap(_create_history_curve).opts(framewise=True)
    panel.extension()

    controls = panel.Column(
        iteration_selector,
        elemental_metric_selector,
        deformation_view_selector,
        history_plot_vars_selector,
    )

    cols = panel.Column(dmap, hist_map)

    graph_row = panel.Row(cols, controls)

    holoviews.save(dmap, filename=r"c:\temp\aaa.gif", fmt='gif')

    _make_animation()

    panel.panel(graph_row).show()




def do_test():
    numpy.random.seed(13)
    xs, ys = numpy.meshgrid(numpy.linspace(-20, 20, 10), numpy.linspace(0, 30, 8))
    xs += xs / 10 + numpy.random.rand(*xs.shape) * 4
    ys += ys / 10 + numpy.random.rand(*ys.shape) * 4

    zs = numpy.arange(80).reshape(8, 10).astype(float)
    zs[3,4] = numpy.nan
    qmesh = holoviews.QuadMesh((xs, ys, zs))

    qmesh.opts(colorbar=True, width=380)


    panel.panel(qmesh).show()


def do_test_poly():
    def make_poly(i):
        x= i
        y = 10+i
        return {
            ('x', 'y'): [(x, y), (x+0.4, y), (x+0.4, y+0.7), (x, y+0.8)],
            'level': i}

    poly_data = [make_poly(i) for i in range(5)]
    polys = holoviews.Polygons(poly_data, vdims='level')
    polys.opts(color='level', line_width=1, padding=0.1)
    panel.panel(polys).show()


def main():
    status_checks = get_status_checks()
    scatter_data = [
        (st.iteration_num, st.metric_val) for st in status_checks
    ]
    scatter = holoviews.Scatter(scatter_data)
    scatter.options(cmap='viridis')
    panel.panel(scatter).show()


if __name__ == '__main__':
    make_dashboard(WORKING_DIR_TEMP)
