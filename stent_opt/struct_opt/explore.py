import collections
import itertools
import pathlib
import typing

import holoviews
import panel
import numpy


from stent_opt.abaqus_model import base, element
from stent_opt.struct_opt import history, design

holoviews.extension('bokeh')

def _get_most_recent_working_dir() -> pathlib.Path:
    subdirs = (p for p in pathlib.Path(r"C:\TEMP\aba").iterdir() if p.is_dir())

    def most_recent(p: pathlib.Path):
        return p.stat().st_ctime

    return max(subdirs, key=most_recent)


WORKING_DIR_TEMP = pathlib.Path(r"E:\Simulations\StentOpt\AA-70")  # _get_most_recent_working_dir() # pathlib.Path(r"E:\Simulations\StentOpt\AA-33")
FIG_SIZE_UNI = (2000, 1350)
FIG_SIZE_LAPTOP = (1200, 750)

FIG_SIZE = FIG_SIZE_UNI

class ContourView(typing.NamedTuple):
    iteration_num: int
    metric_name: str
    deformed: bool


def get_status_checks() -> typing.List["history.StatusCheck"]:
    history_db = history.make_history_db(WORKING_DIR_TEMP)
    with history.History(history_db) as hist:
        return list(hist.get_status_checks(iter_greater_than=0))


def _build_contour_view_data(hist: history.History) -> typing.Iterable[typing.Tuple[ContourView, typing.Dict[int, float]]]:
    """Make the view info for a contour."""

    def make_contour_view(status_check: history.StatusCheck) -> ContourView:
        return ContourView(
            iteration_num=status_check.iteration_num,
            metric_name=status_check.metric_name,
            deformed=None,
        )

    for contour_view, sub_iter in itertools.groupby(hist.get_status_checks(0), make_contour_view):
        elem_vals = {status_check.elem_num: status_check.metric_val for status_check in sub_iter}
        yield contour_view, elem_vals


def make_contour(
        node_pos: typing.Dict[int, base.XYZ],
        contour_view: ContourView,
        element_to_value: typing.Dict[element.Element, float],
) -> holoviews.Polygons:

    """Make a single Holoviews contour."""

    # TODO - try again with QuadMesh - maybe we can do some masking/nan/blah blah blah stuff...

    def gen_poly_data(elem, value):
        one_poly_data = {
            ('x', 'y'): [ (node_pos[iNode].x, node_pos[iNode].y) for iNode in elem.connection],
            'level': value}
        return one_poly_data

    poly_data = [gen_poly_data(elem, value) for elem, value in element_to_value.items()]

    polys = holoviews.Polygons(poly_data, vdims='level', group=contour_view.metric_name)
    polys.opts(color='level', aspect='equal', line_width=0.1, padding=0.1, width=FIG_SIZE[0], height=FIG_SIZE[1], invert_axes=True)

    return polys


def make_quadmesh(
        stent_params: design.StentParams,
        active_elements: typing.FrozenSet[design.PolarIndex],
        node_idx_pos: typing.Dict[design.PolarIndex, base.XYZ],
        contour_view: ContourView,
        elem_idx_to_value: typing.Dict[design.PolarIndex, float],
) -> holoviews.Overlay:
    """Make a single Quadmesh representation"""

    # If it's not deformed, all the nodal positions should be there so we can plot inactive elements as well.
    INCLUDE_GHOST_ELEMENENTS = not contour_view.deformed

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

        elif INCLUDE_GHOST_ELEMENENTS:
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

    qmesh_real = holoviews.QuadMesh((X, Y, Z), vdims='level', group=contour_view.metric_name)
    qmesh_real.options(cmap='viridis')

    qmesh_list = [qmesh_real]
    if INCLUDE_GHOST_ELEMENENTS and not all_ghosts_nan:
        qmesh_ghost = holoviews.QuadMesh((X, Y, Z_ghost), vdims='level', group=contour_view.metric_name)
        qmesh_ghost.options(cmap='inferno')
        qmesh_list.append(qmesh_ghost)

    for qmesh in qmesh_list:
        qmesh.opts(aspect='equal', line_width=0.1, padding=0.1, width=FIG_SIZE[0], height=FIG_SIZE[1], colorbar=True)

    return holoviews.Overlay(qmesh_list)


def make_dashboard(working_dir: pathlib.Path):
    # Get the data from the
    print(working_dir)

    history_db = history.make_history_db(working_dir)
    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        snapshots = {snap.iteration_num: snap for snap in hist.get_snapshots()}

        # Node undeformed position and connectivity
        node_pos_undeformed = {iNode: pos.to_xyz() for iNode, idx, pos in design.generate_nodes(stent_params)}
        node_pos_deformed_all_iters = collections.defaultdict(dict)
        node_num_to_idx = {iNode: idx for iNode, idx, pos in design.generate_nodes(stent_params)}
        for node_pos in hist.get_node_positions():
            node_pos_deformed_all_iters[node_pos.iteration_num][node_pos.node_num] = base.XYZ(x=node_pos.x, y=node_pos.y, z=node_pos.z)

        num_to_elem = {iElem: elem for idx, iElem, elem in design.generate_stent_part_elements(stent_params)}
        num_to_idx = {iElem: idx for idx, iElem, elem in design.generate_stent_part_elements(stent_params)}

        plot_dict = {}
        for contour_view_raw, elem_data in _build_contour_view_data(hist):

            for deformed, node_pos in (
                    (False, node_pos_undeformed),
                    (True, node_pos_deformed_all_iters[contour_view_raw.iteration_num]),

            ):
                contour_view = contour_view_raw._replace(deformed=deformed)
                active_elements = frozenset(num_to_idx[iElem] for iElem in snapshots[contour_view.iteration_num].active_elements)
                print(contour_view)

                # Some measures (like RegionGradient) can exist even if the element is inactive. Strip these out of the display.
                elem_to_value = {
                    num_to_elem[iElem]: val for
                    iElem, val in elem_data.items()
                    if iElem in snapshots[contour_view.iteration_num].active_elements}

                elem_idx_to_value = {
                    num_to_idx[iElem]: val for
                    iElem, val in  elem_data.items()}

                node_idx_pos = {node_num_to_idx[iNode]: pos for iNode, pos in node_pos.items()}

                qmesh = make_quadmesh(stent_params, active_elements, node_idx_pos, contour_view, elem_idx_to_value)
                plot_dict[contour_view] = qmesh

        hmap = holoviews.HoloMap(plot_dict, kdims=list(contour_view._fields))
        panel.panel(hmap).show()


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
