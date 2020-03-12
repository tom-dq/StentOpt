# Utilities for separating the rigid body and deformation gradients of some point cloud.
import math
import typing

import numpy
import scipy
import scipy.optimize
import rmsd

from stent_opt.abaqus_model.base import XYZ

T_NodeKey = typing.TypeVar("T_NodeKey")
T_NodeMap = typing.Dict[T_NodeKey, XYZ]

class RigidMotion(typing.NamedTuple):
    R: numpy.array  # Rotation matrix, 3x3
    t: numpy.array  # Deformation vector.


def _get_points_array(key_order, nodes: T_NodeMap) -> numpy.array:
    points_in_order = tuple(nodes[k] for k in key_order)
    points_array = numpy.array(points_in_order)
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

    return _get_rigid_motion_sorkine_hornung(p_orig, q_deformed)


def _get_rigid_motion_sorkine_hornung(p_orig: numpy.array, q_deformed: numpy.array) -> RigidMotion:
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


def _get_rigid_motion_horn_1987(p_orig: numpy.array, q_deformed: numpy.array) -> RigidMotion:
    """
    This seems to be working well - uses quaternions and is robust.

    Eggert, David W., Adele Lorusso, and Robert B. Fisher. "Estimating 3-D rigid body transformations: a comparison of four major algorithms." Machine vision and applications 9.5-6 (1997): 272-290.
    Algorithm 3.3
    """

    # Get the centroids
    p_bar = numpy.mean(p_orig, axis=0)
    q_bar = numpy.mean(q_deformed, axis=0)

    # Centre the vectors
    p_centered = p_orig - p_bar
    q_centered = q_deformed - q_bar

    def s_matrix_term(a_ord: int, b_ord: int) -> float:
        """Get a term in S_ab"""
        m_c_i = p_centered[:, a_ord]
        d_c_i = q_centered[:, b_ord]
        return numpy.dot(m_c_i, d_c_i)


    x, y, z = 0, 1, 2  # convenience!
    S = numpy.zeros(shape=(3, 3))
    for a_ord in [0, 1, 2]:
        for b_ord in [0, 1, 2]:
            S[a_ord, b_ord] = s_matrix_term(a_ord, b_ord)

    # Make P from the S terms
    P = numpy.array([
        [S[x, x] + S[y, y] + S[z, z],   S[y, z] - S[z, y],   S[z, x] - S[x, z],  S[x, y] - S[y, x]  ],
        [S[y, z] - S[z, y],   S[x, x] - S[y, y] - S[z, z],   S[x, y] + S[y, z],  S[z, x] + S[x, z]  ],
        [S[z, x] - S[x, z],  S[x, y] + S[y, x],  S[y, y] - S[x, x] - S[z, z],    S[y, z] + S[z, y]  ],
        [S[x, y] - S[y, x],  S[z, x] + S[x, z],  S[y, z] + S[z, y],  S[z, z] - S[x, x] - S[y, y]]
    ])

    # Get the eigenvector associated with the maximum eigenvalue.
    eig_vals, eig_vects = numpy.linalg.eig(P)
    val_and_idx = [(eig_val, idx) for idx, eig_val in enumerate(eig_vals)]
    val_and_idx.sort(reverse=True)
    max_eig_idx = val_and_idx[0][1]
    q_hat = eig_vects[:, max_eig_idx]

    # Generate the rotation matrix
    q0, q1, q2, q3 = q_hat[:]

    R = numpy.array([
        [q0**2 + q1**2 - q2**2 - q3**2,   2*(q1*q2 - q0*q3),   2*(q1*q3 + q0*q2)],
        [2*(q2*q1 + q0*q3),   q0**2 - q1**2 + q2**2 - q3**2,  2*(q2*q3 - q0*q1)],
        [2*(q3*q1 - q0*q2),   2*(q3*q2 + q0*q1),   q0**2 - q1**2 - q2**2 + q3**2],
    ])

    t = q_bar - R @ p_bar

    return RigidMotion(R=R, t=t)



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

def _DEBUG_plot_rigid_points(node_keys, p_orig, p_with_rigid, q_deformed):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for iPoint in range(p_orig.shape[0]):

        def plot_one_point(vect, style):
            x, y, z = vect[iPoint, 0], vect[iPoint, 1], vect[iPoint, 2]
            ax.plot([x], [y], [z], style)
            ax.text(x, y, z, str(node_keys[iPoint]))

        # Orig
        plot_one_point(p_orig, 'kx')
        plot_one_point(p_with_rigid, 'bx')
        plot_one_point(q_deformed, 'bo')

    plt.show()


def nodal_deformation(nominal_length: float, orig: T_NodeMap, deformed: T_NodeMap) -> float:
    """Gets some normalised deformation measure for the subset of nodes overall."""

    node_keys = tuple(sorted(set.union(set(orig.keys()), set(deformed.keys()))))

    p_orig = _get_points_array(node_keys, orig)
    q_deformed = _get_points_array(node_keys, deformed)

    rigid_motion = _get_rigid_motion_horn_1987(p_orig, q_deformed)
    p_with_rigid = _apply_rigid_motion(rigid_motion, p_orig)

    deformation_only = q_deformed - p_with_rigid
    norms = numpy.linalg.norm(deformation_only, axis=1)

    # TESTING CODE
    # n_vect = numpy.array([norms]).T
    # TESTING = numpy.hstack((p_orig, q_deformed, n_vect))
    # print(TESTING)
    # END TESTING CODE

    if len(norms) != len(orig):
        raise ValueError("Looks like Tom got an index/axis wrong someplace...")

    deformation_ratio = float(numpy.mean(norms) / nominal_length)

    DEBUG_MAKE_PLOT = False
    if DEBUG_MAKE_PLOT:
        _DEBUG_plot_rigid(p_orig, p_with_rigid, q_deformed)
        #_DEBUG_plot_rigid_points(node_keys, p_orig, p_with_rigid, q_deformed)

    return deformation_ratio


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


def _euler_angle_rotation_matrix(alpha: float, beta: float, gamma: float) -> numpy.array:
    """Generate the 3x3 rotation matrix from the X,Z,X euler angle rotations.
    https://en.wikipedia.org/wiki/Euler_angles"""

    s1, c1 = math.sin(alpha), math.cos(alpha)
    s2, c2 = math.sin(beta), math.cos(beta)
    s3, c3 = math.sin(gamma), math.cos(gamma)

    R = numpy.array([
        [c2, -1.0*c3*s2, s2*s3],
        [c1*s2, c1*c2*c3 - s1*s3, -1.0*c3*s1 - c1*c2*s3],
        [s1*s2, c1*s3 + c2*c3*s1, c1*c3 - c2*s1*s3],
    ])

    return R


def nodal_deformation_least_squares(nominal_length: float, orig: T_NodeMap, deformed: T_NodeMap) -> float:
    """Gets some normalised deformation measure for the subset of nodes overall."""

    node_keys = tuple(sorted(set.union(set(orig.keys()), set(deformed.keys()))))

    p_orig = _get_points_array(node_keys, orig)
    q_deformed = _get_points_array(node_keys, deformed)

    # Step 1 - get the centroids
    p_bar = numpy.mean(p_orig, axis=0)
    q_bar = numpy.mean(q_deformed, axis=0)
    t_init = q_bar - p_bar

    def f_to_minimise(x):
        """x[0:3] -> euler angle rotation.
        x[3:6] -> translation"""

        rigid_motion = RigidMotion(
            R=_euler_angle_rotation_matrix(x[0], x[1], x[2]),
            t=x[3:6]
        )

        p_with_rigid = _apply_rigid_motion(rigid_motion, p_orig)
        deformation_only_working = q_deformed - p_with_rigid
        deformation_norm_working = numpy.linalg.norm(deformation_only_working, axis=1)
        return deformation_norm_working


    # Initial guess
    x0 = numpy.concatenate([numpy.array([0.0, 0.0, 0.0]), t_init])

    opt_res = scipy.optimize.least_squares(f_to_minimise, x0, method='lm')

    print(opt_res)

    deformation_norm = f_to_minimise(opt_res.x)
    if len(deformation_norm) != len(orig):
        raise ValueError("Looks like Tom got an index/axis wrong someplace...")

    return float(numpy.mean(deformation_norm) / nominal_length)


def nodal_deformation_rmsd(nominal_length: float, orig: T_NodeMap, deformed: T_NodeMap) -> float:
    """Use this: https://github.com/charnley/rmsd"""

    node_keys = tuple(sorted(set.union(set(orig.keys()), set(deformed.keys()))))

    p_orig = _get_points_array(node_keys, orig)
    q_deformed = _get_points_array(node_keys, deformed)

    # Step 1 - get the centroids
    p_bar = numpy.mean(p_orig, axis=0)
    q_bar = numpy.mean(q_deformed, axis=0)

    # Centre both of them
    p_cent = p_orig - numpy.mean(p_orig, axis=0)
    q_cent = q_deformed - numpy.mean(q_deformed, axis=0)

    U = rmsd.kabsch(p_cent, q_cent)
    p_cent_rotate = numpy.dot(p_cent, U)

    deformation_only = q_cent - p_cent_rotate
    deformation_norm = numpy.linalg.norm(deformation_only, axis=1)

    DEBUG_MAKE_PLOT = False
    if DEBUG_MAKE_PLOT:
        _DEBUG_plot_rigid(p_cent, p_cent_rotate, q_cent)


    return float(numpy.mean(deformation_norm) / nominal_length)



if __name__ == "__main__":
    test_from_real_data_should_be_small()

if __name__ == "__main2__":
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

    #print(nodal_deformation_rmsd(1.0, orig_points, move_in_z))
    print(nodal_deformation(1.0, orig_points, rotate_90_deg))
    #print(nodal_deformation_rmsd(1.0, orig_points, rotate_90_deg))




















