# Utilities for separating the rigid body and deformation gradients of some point cloud.

import typing

import numpy
import scipy

from stent_opt.abaqus_model.base import XYZ

T_NodeKey = typing.TypeVar("T_NodeKey")
T_NodeMap = typing.Dict[T_NodeKey, XYZ]

class RigidMotion(typing.NamedTuple):
    R: numpy.array  # Rotation matrix, 3x3
    t: numpy.array  # Deformation vector.


def _get_points_array(key_order, nodes: T_NodeMap) -> numpy.array:
    points_in_order = (nodes[k] for k in key_order)
    points_raw = tuple(sorted(points_in_order))
    points_array = numpy.array(points_raw)
    return points_array


def _get_node_map(key_order, points: numpy.array) -> T_NodeMap:
    out_points = {}
    for idx, k in enumerate(key_order):
        x, y, z = [float(_x) for _x in points[idx, :]]
        out_points[k] = XYZ(x, y, z)

    return out_points

def get_rigid_motion(orig: T_NodeMap, deformed: T_NodeMap) -> RigidMotion:
    # Assume all points are weighted equally, and put them in the same order.
    node_keys = tuple(sorted(set.union(set(orig.keys()), set(deformed.keys()))))

    p_orig = _get_points_array(node_keys, orig)
    q_deformed = _get_points_array(node_keys, deformed)

    return _get_rigid_motion(p_orig, q_deformed)


def _get_rigid_motion(p_orig: numpy.array, q_deformed: numpy.array) -> RigidMotion:
    """Least-Squares Rigid Motion Using SVD
    Olga Sorkine-Hornung and Michael Rabinovich
    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    """

    d = 3

    # Step 1 - get the centroids
    p_bar = numpy.mean(p_orig, axis=0)
    q_bar = numpy.mean(q_deformed, axis=0)

    # Step 2 - centred vectors
    x = p_orig - p_bar
    y = q_deformed - q_bar

    # Step 3 - compute the covariance matrix
    S = x.T @ y

    # Step 4 - SVD
    u, s, vh = numpy.linalg.svd(S, full_matrices=True)

    # Change from published algorithm - don't do the determinant stuff (so, never flipping).
    # middle_bit = numpy.eye(d)
    # det_vut = numpy.linalg.det(vh @ u.T)
    # middle_bit[d-1, d-1] = det_vut
    # R = vh @ middle_bit @ u.T
    R = vh @ u.T

    t = q_bar - R @ p_bar

    return RigidMotion(R=R, t=t)

    # TODO - make this better - I don't think it's getting the best fit at the moment. OpenCV, for example, uses RANSAC
    #   see estimateRigidTransform here: https://github.com/opencv/opencv/blob/324851882ab95e02bb17ea3e67b80f6af677c9f8/modules/video/src/lkpyramid.cpp#L1490


def apply_rigid_motion(rigid_motion: RigidMotion, orig: T_NodeMap) -> T_NodeMap:
    """Apply the rigid body motion to some undeformed nodes."""

    node_keys = tuple(sorted(orig.keys()))
    p_orig = _get_points_array(node_keys, orig)
    p_def = _apply_rigid_motion(rigid_motion, p_orig)
    return _get_node_map(node_keys, p_def)


def _apply_rigid_motion(rigid_motion: RigidMotion, p_orig: numpy.array) -> numpy.array:
    # trans_only = p_orig + rigid_motion.t
    # p_def = rigid_motion.R @ trans_only.T
    # return p_def.T
    rot_only = rigid_motion.R @ p_orig.T
    p_def = rot_only.T + rigid_motion.t
    return p_def


def _DEBUG_plot_rigid(p_orig, p_with_rigid, q_deformed):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for iPoint in range(p_orig.shape[0]):

        line_rigid = numpy.zeros( (2, 3))
        line_pure_def = numpy.zeros( (2, 3))

        line_rigid[0, :] = p_orig[iPoint, :]
        line_rigid[1, :] = p_with_rigid[iPoint, :]
        line_pure_def[0, :] = p_with_rigid[iPoint, :]
        line_pure_def[1, :] = q_deformed[iPoint, :]

        ax.plot(*line_rigid.T.tolist(), color='k', linewidth=0.8)
        ax.plot(*line_pure_def.T.tolist(), color='b')

    plt.show()


def nodal_deformation(nominal_length: float, orig: T_NodeMap, deformed: T_NodeMap) -> float:
    """Gets some normalised deformation measure for the subset of nodes overall."""

    node_keys = tuple(sorted(set.union(set(orig.keys()), set(deformed.keys()))))

    p_orig = _get_points_array(node_keys, orig)
    q_deformed = _get_points_array(node_keys, deformed)

    rigid_motion = _get_rigid_motion(p_orig, q_deformed)
    p_with_rigid = _apply_rigid_motion(rigid_motion, p_orig)

    deformation_only = q_deformed - p_with_rigid
    norms = numpy.linalg.norm(deformation_only, axis=1)
    if len(norms) != len(orig):
        raise ValueError("Looks like Tom got an index/axis wrong someplace...")


    DEBUG_MAKE_PLOT = True
    if DEBUG_MAKE_PLOT:
        _DEBUG_plot_rigid(p_orig, p_with_rigid, q_deformed)

    return float(numpy.mean(norms) / nominal_length)


def test_from_real_data_should_be_small():
    orig = {2850: XYZ(x=0.30515651581082903, y=2.7085427135678395, z=0.5739159353583025),
            2851: XYZ(x=0.30515651581082903, y=2.7638190954773867, z=0.5739159353583025),
            2852: XYZ(x=0.30515651581082903, y=2.819095477386935, z=0.5739159353583025),
            2853: XYZ(x=0.30515651581082903, y=2.874371859296483, z=0.5739159353583025),
            2854: XYZ(x=0.30515651581082903, y=2.92964824120603, z=0.5739159353583025),
            3050: XYZ(x=0.32499999999999996, y=2.7085427135678395, z=0.5629165124598852),
            3051: XYZ(x=0.32499999999999996, y=2.7638190954773867, z=0.5629165124598852),
            3052: XYZ(x=0.32499999999999996, y=2.819095477386935, z=0.5629165124598852),
            3053: XYZ(x=0.32499999999999996, y=2.874371859296483, z=0.5629165124598852),
            3054: XYZ(x=0.32499999999999996, y=2.92964824120603, z=0.5629165124598852),
            3250: XYZ(x=0.3444475217515832, y=2.7085427135678395, z=0.5512312625016769),
            3251: XYZ(x=0.3444475217515832, y=2.7638190954773867, z=0.5512312625016769),
            3252: XYZ(x=0.3444475217515832, y=2.819095477386935, z=0.5512312625016769),
            3253: XYZ(x=0.3444475217515832, y=2.874371859296483, z=0.5512312625016769),
            3254: XYZ(x=0.3444475217515832, y=2.92964824120603, z=0.5512312625016769),
            9050: XYZ(x=0.3521036720894181, y=2.7085427135678395, z=0.6622106946441952),
            9051: XYZ(x=0.3521036720894181, y=2.7638190954773867, z=0.6622106946441952),
            9052: XYZ(x=0.3521036720894181, y=2.819095477386935, z=0.6622106946441952),
            9053: XYZ(x=0.3521036720894181, y=2.874371859296483, z=0.6622106946441952),
            9054: XYZ(x=0.3521036720894181, y=2.92964824120603, z=0.6622106946441952),
            9250: XYZ(x=0.37499999999999994, y=2.7085427135678395, z=0.649519052838329),
            9251: XYZ(x=0.37499999999999994, y=2.7638190954773867, z=0.649519052838329),
            9252: XYZ(x=0.37499999999999994, y=2.819095477386935, z=0.649519052838329),
            9253: XYZ(x=0.37499999999999994, y=2.874371859296483, z=0.649519052838329),
            9254: XYZ(x=0.37499999999999994, y=2.92964824120603, z=0.649519052838329),
            9450: XYZ(x=0.3974394481749037, y=2.7085427135678395, z=0.6360360721173195),
            9451: XYZ(x=0.3974394481749037, y=2.7638190954773867, z=0.6360360721173195),
            9452: XYZ(x=0.3974394481749037, y=2.819095477386935, z=0.6360360721173195),
            9453: XYZ(x=0.3974394481749037, y=2.874371859296483, z=0.6360360721173195),
            9454: XYZ(x=0.3974394481749037, y=2.92964824120603, z=0.6360360721173195)}

    deformed = {2850: XYZ(x=1.3744884133338928, y=2.9893102645874023, z=2.434308886528015),
                2851: XYZ(x=1.3740331530570984, y=3.0443387031555176, z=2.428528308868408),
                2852: XYZ(x=1.3735584616661072, y=3.099322646856308, z=2.4227144718170166),
                2853: XYZ(x=1.3731067776679993, y=3.154302328824997, z=2.416901111602783),
                2854: XYZ(x=1.3726399540901184, y=3.2092836499214172, z=2.4110978841781616),
                3050: XYZ(x=1.3945273756980896, y=2.9883379340171814, z=2.423520088195801),
                3051: XYZ(x=1.3940618634223938, y=3.043363332748413, z=2.41767954826355),
                3052: XYZ(x=1.3936216235160828, y=3.098342001438141, z=2.411882162094116),
                3053: XYZ(x=1.3931577801704407, y=3.153321325778961, z=2.4060587882995605),
                3054: XYZ(x=1.3927127718925476, y=3.208304852247238, z=2.400255560874939),
                3250: XYZ(x=1.414108544588089, y=2.987268805503845, z=2.4119333028793335),
                3251: XYZ(x=1.4136832058429718, y=3.0422960221767426, z=2.4061323404312134),
                3252: XYZ(x=1.4132353365421295, y=3.0972809195518494, z=2.400308132171631),
                3253: XYZ(x=1.4127823412418365, y=3.1522598266601562, z=2.3944867849349976),
                3254: XYZ(x=1.4123353064060211, y=3.2072407603263855, z=2.388680577278137),
                9050: XYZ(x=1.4206113517284393, y=2.9989163875579834, z=2.5220757722854614),
                9051: XYZ(x=1.420164316892624, y=3.0539737343788147, z=2.516286611557007),
                9052: XYZ(x=1.4197095334529877, y=3.108986347913742, z=2.5104880332946777),
                9053: XYZ(x=1.4192661941051483, y=3.1639652848243713, z=2.50467312335968),
                9054: XYZ(x=1.418807476758957, y=3.218946009874344, z=2.4988518953323364),
                9250: XYZ(x=1.4436103105545044, y=2.997778505086899, z=2.5095334649086),
                9251: XYZ(x=1.4431841373443604, y=3.052841067314148, z=2.503766715526581),
                9252: XYZ(x=1.4427326917648315, y=3.1078494787216187, z=2.497955858707428),
                9253: XYZ(x=1.4422786235809326, y=3.1628313064575195, z=2.4921409487724304),
                9254: XYZ(x=1.4418190717697144, y=3.2178133130073547, z=2.486320197582245),
                9450: XYZ(x=1.4662100970745087, y=2.9965653717517853, z=2.4963013529777527),
                9451: XYZ(x=1.465796798467636, y=3.0516233444213867, z=2.4905003905296326),
                9452: XYZ(x=1.4653571546077728, y=3.1066353917121887, z=2.4846875071525574),
                9453: XYZ(x=1.4649040400981903, y=3.1616156697273254, z=2.4788851141929626),
                9454: XYZ(x=1.4644388854503632, y=3.2165971398353577, z=2.47306090593338)}

    NOMINAL_LENGTH = 0.2
    measure = nodal_deformation(NOMINAL_LENGTH, orig, deformed)
    print(measure)



if __name__ == "__main__":
    test_from_real_data_should_be_small()

if False:

    orig_points = {
        "a": XYZ(0.0, 0.0, 0.0),
        "b": XYZ(1.0, 0.0, 0.0),
        "c": XYZ(1.0, 1.0, 0.0),
        "d": XYZ(0.0, 1.0, 0.0),
        "e": XYZ(0.5, 0.5, 0.0),
    }

    move_in_z = {
        "a": XYZ(0.0, 0.0, 1.0),
        "b": XYZ(1.0, 0.0, 1.0),
        "c": XYZ(1.0, 1.0, 1.0),
        "d": XYZ(0.0, 1.0, 1.0),
        "e": XYZ(0.5, 0.5, 1.0),
    }

    strain_in_x = {
        "a": XYZ(0.0, 0.0, 0.0),
        "b": XYZ(2.0, 0.0, 0.0),
        "c": XYZ(2.0, 1.0, 0.0),
        "d": XYZ(0.0, 1.0, 0.0),
        "e": XYZ(1.0, 0.5, 0.0),
    }

    rotate_90_deg = {
        "a": XYZ(0.0, 0.0, 0.0),
        "b": XYZ(0.0, 1.0, 0.0),
        "c": XYZ(-1.0, 1.0, 0.0),
        "d": XYZ(-1.0, 0.0, 0.0),
        "e": XYZ(-0.5, 0.5, 0.0),
    }

    print(nodal_deformation(1.0, orig_points, move_in_z))
    print(nodal_deformation(1.0, orig_points, strain_in_x))
    print(nodal_deformation(1.0, orig_points, rotate_90_deg))




















