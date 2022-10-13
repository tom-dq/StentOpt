import collections
import typing
import colorsys

import matplotlib as mpl
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as mcolors
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import stent_opt.struct_opt.score
from stent_opt.struct_opt import design, generation
from stent_opt.abaqus_model import base


class ViewAngle(typing.NamedTuple):
    elev: float
    azim: float


_VIEW_SIDE = ViewAngle(elev=17.0, azim=-20.0)
_VIEW_TOP = ViewAngle(elev=60.0, azim=0.0)
_VIEW_2D = ViewAngle(elev=90.0, azim=0.0)
VIEW_ANGLE = _VIEW_2D


class BackgroundColour:
    # Keeps the same background colour for each type of plot.
    _existing: dict
    _unused_raw_cols: list

    def __init__(self):
        self._existing = dict()
        self._unused_raw_cols = list(reversed(mcolors.TABLEAU_COLORS))

    def get_colour(self, comp_name):
        if comp_name not in self._existing:
            raw_col = self._unused_raw_cols.pop()
            lighter_col = self.lighten_color(raw_col)
            self._existing[comp_name] = lighter_col

        return self._existing[comp_name]

    @staticmethod
    def lighten_color(color):
        # https://stackoverflow.com/a/49601444/11308690
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """

        AMOUNT = 1.7

        try:
            c = mcolors.cnames[color]

        except:
            c = color

        c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, AMOUNT * c[1])), c[2])


BACKGROUND_COLOUR_TRACKER = BackgroundColour()


def render_status(
    stent_design: "design.StentDesign",
    node_position: typing.Dict[int, base.XYZ],
    components: typing.Iterable[
        typing.Union[
            "generation.PrimaryRankingComponent", "generation.SecondaryRankingComponent"
        ]
    ],
    title_prefix: str,
):

    components = list(components)

    # Make sure there's only one (for now!)
    comp_types = {comp.comp_name for comp in components}
    if len(comp_types) != 1:
        raise ValueError(comp_types)

    comp_type = comp_types.pop()

    # Get all the faces of the stent elements
    def faces_and_values():
        elem_num_to_indices = {
            iElem: idx
            for iElem, idx in design.generate_elem_indices(
                stent_design.stent_params.divs
            )
        }
        elem_lookup = {
            elem_num_to_indices[comp.elem_id]: comp.value for comp in components
        }

        for one_elem in stent_design.active_elements:
            elem_connection = design.get_c3d8_connection(
                stent_design.stent_params.divs, one_elem
            )
            for one_face_idx in stent_opt.struct_opt.score.FACES_OF[
                stent_design.stent_params.stent_element_type
            ]:
                global_node_nums = [elem_connection[i] for i in one_face_idx]
                yield global_node_nums, elem_lookup[one_elem]

    # Find only the exposed faces.
    faces_paired = collections.defaultdict(list)
    for global_node_nums, value in faces_and_values():
        sorted_node_nums = tuple(sorted(global_node_nums))
        faces_paired[sorted_node_nums].append((global_node_nums, value))

    single_faces_and_vals = [
        one_list[0] for one_list in faces_paired.values() if len(one_list) == 1
    ]

    # Render it
    poly_list = []
    for face_nodes, val in single_faces_and_vals:
        verts = [node_position[iNode] for iNode in face_nodes]

        poly_list.append((verts, val))

    bg_col = BACKGROUND_COLOUR_TRACKER.get_colour(comp_type)

    show_polygons(f"{title_prefix} {comp_type}", poly_list, bg_col)


def show_design(stent_design: "design.StentDesign"):
    # This is for 2D designs

    # Oh gawd but this is a one off I swear
    from stent_opt.struct_opt.design import generate_nodes, generate_stent_part_elements

    node_position = {
        iNode: xyz.to_xyz()
        for iNode, polar_index, xyz in generate_nodes(stent_design.stent_params)
    }

    poly_list = []
    for idx, iElem, elem in generate_stent_part_elements(stent_design.stent_params):
        if idx in stent_design.active_elements:
            verts = [node_position[iNode] for iNode in elem.connection]
            poly_list.append((verts, 1.0))

    show_polygons(f"show_design", poly_list, (0.9, 0.9, 0.9), dims=2)


def show_polygons(title, poly_list, bg_col, dims: int):

    if dims == 3:
        is_3d = True
        ax = a3.Axes3D(pl.figure(figsize=(10.5, 7)))
        fig = ax.figure

    elif dims == 2:
        is_3d = False
        fig = plt.figure(figsize=(10.5, 7))
        ax = fig.gca()

    else:
        raise ValueError(dims)

    min_c = min(val for _, val in poly_list)
    max_c = max(val for _, val in poly_list)

    if max_c - min_c < 1e-10:
        max_c = max_c + 1e-6

    norm = mpl.colors.Normalize(vmin=min_c, vmax=max_c)
    cmap = cm.plasma

    colour_mapping = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Get the limits manually
    def get_ordinate(i):
        for vert_list, _ in poly_list:
            for v in vert_list:
                yield v[i]

    def get_lim(i):
        vals = list(get_ordinate(i))
        return min(vals), max(vals)

    pl.title(title)
    all_verts = [vert for vert, _ in poly_list]
    all_colours = [colour_mapping.to_rgba(val) for _, val in poly_list]

    if is_3d:
        tri = a3.art3d.Poly3DCollection(all_verts)

    else:
        all_verts_2d = []
        for one_poly in all_verts:
            verts_2d = [(x, y) for x, y, z in one_poly]
            all_verts_2d.append(verts_2d)
        tri = PolyCollection(all_verts_2d)

    tri.set_facecolors(all_colours)
    # tri.set_edgecolor('k')

    ax.set_facecolor(bg_col)
    # fig.patch.set_facecolor(bg_col)

    if is_3d:
        ax.add_collection3d(tri)
    else:
        ax.add_collection(tri)

    # Colourbar
    axins = inset_axes(
        ax,
        width="75%",  # width = 50% of parent_bbox width
        height="2%",  # height : 5%
        loc="lower center",
        bbox_to_anchor=(0, 0.05, 1, 1),
        bbox_transform=ax.transAxes,
    )

    fig.colorbar(colour_mapping, cax=axins, orientation="horizontal")

    if is_3d:
        ax.set_xlim3d(*get_lim(0))
        ax.set_ylim3d(*get_lim(1))
        ax.set_zlim3d(*get_lim(2))

    else:
        ax.set_xlim(*get_lim(0))
        ax.set_ylim(*get_lim(1))

    if is_3d:
        # Hide the 3D axis
        ax.set_axis_off()
        ax.azim = VIEW_ANGLE.azim
        ax.elev = VIEW_ANGLE.elev

    pl.show()


if __name__ == "__main__":
    render_thing()
