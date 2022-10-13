# This is run by Abaqus Python and populates the output odb.
# abaqus cae noGui=odb_extract.py -- c:\temp\db.db
# The double dash is for arguments passed to the script but ignored by Abaqus itself.

# Rule of thumb - make everything a double!
import itertools
import math

SAVE_IMAGES = False

import sys
import collections

import abaqusConstants
import odbAccess
import numpy

if SAVE_IMAGES:
    import abaqus


EXTRACT_ALL_STEPS = True  # If False, only get the last frame of each step.

STEP_IDX_EXPAND = 0
STEP_IDX_RELEASE_OSCILLATE = 1

# From here just for testing
# https://www.engineeringtoolbox.com/steel-endurance-limit-d_1781.html
TEMP_SIGMA_UTS = 540.0
TEMP_SIGMA_ENDURANCE = 270.0

from datastore import Datastore
from db_defs import (
    Frame,
    NodePos,
    NodeReact,
    ElementStress,
    ElementPEEQ,
    ElementEnergyElastic,
    ElementEnergyPlastic,
    ElementFatigueResult,
    ElementNodeForces,
    HistoryResult,
    expected_history_results,
)

# Get the command line option (should be last!).
fn_odb = sys.argv[-4]
db_fn = sys.argv[-3]
_override_z_val_str = sys.argv[-2]
add_initial_node_pos_str = sys.argv[-1]

if add_initial_node_pos_str == str(True):
    add_initial_node_pos = True

elif add_initial_node_pos_str == str(False):
    add_initial_node_pos = False

else:
    raise ValueError(add_initial_node_pos_str)


# If we are doing a planar analysis and we put the nodes at z=NomRadius, in the .odb file they're
# back at the origin. Since we depend on them being at real z for later post-processing, put them back.
if _override_z_val_str == str(None):
    override_z_val = None

else:
    override_z_val = float(_override_z_val_str)


# fn_odb = r"C:\TEMP\aba\stent-36.odb"
# db_fn = r"C:\TEMP\aba\db-9.db"

ExtractionMeta = collections.namedtuple(
    "ExtractionMeta",
    (
        "frame",
        "node_labels",
        "elem_labels",
        "node_init_pos",
        "odb_instance",
        "nodes_enforced_displacement",
    ),
)


def print_in_term(x):
    # Old Python Syntax!
    print >> sys.__stdout__, x


def _get_nodes_theta_max(this_odb):
    def produce_node_numbers():
        NAME_START = "STENT-SET_PLANARSTENTTHETAMAX"
        for node_set_name, node_set in this_odb.rootAssembly.nodeSets.items():
            if node_set_name.startswith(NAME_START):
                for nodes in node_set.nodes:
                    for one_node in nodes:
                        yield one_node.label

    return set(produce_node_numbers())


def _get_nodes_on_elements(
    elements,
):
    all_nodes = set()
    for elem in elements:
        all_nodes.update(elem.connectivity)

    return frozenset(all_nodes)


def _get_node_initial_pos(override_z_val, instance):
    out_dict = {}
    for one_node in instance.nodes:
        coords_working = one_node.coordinates.astype("d")

        if override_z_val is not None:
            coords_working[2] = override_z_val

        out_dict[one_node.label] = coords_working

    return out_dict


def _get_data_array_as_double(one_value):
    if one_value.precision == abaqusConstants.SINGLE_PRECISION:
        return one_value.data.astype("d")

    elif one_value.precision == abaqusConstants.DOUBLE_PRECISION:
        return one_value.dataDouble

    else:
        raise TypeError(one_value.precision)


def get_history_results(this_odb):
    """
    :return type: Iterable[HistoryResult]
    """

    fn_odb = this_odb.name

    for step_num, (step_name, one_step) in enumerate(this_odb.steps.items()):
        for hr_name, history_region in one_step.historyRegions.items():
            for hist_res_id in expected_history_results:
                history_regions_output = history_region.historyOutputs[hist_res_id]
                for hist_time, history_value in history_regions_output.data:
                    yield HistoryResult(
                        rowid=None,
                        fn_odb=fn_odb,
                        step_num=step_num,
                        step_name=step_name,
                        history_region=hr_name,
                        history_identifier=hist_res_id,
                        simulation_time=one_step.totalTime + hist_time,
                        history_value=history_value,
                    )


def walk_file_frames(extract_all_steps, this_odb, override_z_val):
    """
    :return type: Iterable[Frame, ExtractionMeta]
    """
    fn_odb = this_odb.name

    nodes_enforced_displacement = _get_nodes_theta_max(this_odb)

    for one_instance_name, instance in this_odb.rootAssembly.instances.items():
        this_part_elem_labels = frozenset(elem.label for elem in instance.elements)
        this_part_node_labels = _get_nodes_on_elements(instance.elements)
        this_part_node_initial_pos = _get_node_initial_pos(override_z_val, instance)
        for step_num, (step_name, one_step) in enumerate(this_odb.steps.items()):

            last_frame_id_of_step = len(one_step.frames) - 1

            for frame_id, frame in enumerate(one_step.frames):
                is_last_step = frame_id == last_frame_id_of_step
                should_yield = extract_all_steps or is_last_step
                if should_yield:
                    yield (
                        Frame(
                            rowid=None,
                            fn_odb=fn_odb,
                            instance_name=one_instance_name,
                            step_num=step_num,
                            step_name=step_name,
                            frame_id=frame_id,
                            frame_value=frame.frameId,
                            simulation_time=one_step.totalTime + frame.frameValue,
                        ),
                        ExtractionMeta(
                            frame=frame,
                            node_labels=this_part_node_labels,
                            elem_labels=this_part_elem_labels,
                            node_init_pos=this_part_node_initial_pos,
                            odb_instance=instance,
                            nodes_enforced_displacement=nodes_enforced_displacement,
                        ),
                    )

                if is_last_step and SAVE_IMAGES:
                    # Print to file?
                    fn = r"c:\temp\aba\Mod-{0}-{1}-{2}.png".format(
                        one_instance_name, step_name, frame_id
                    )
                    abaqus.session.printOptions.setValues(vpBackground=True)
                    abaqus.session.printToFile(fn, format=abaqusConstants.PNG)


def get_stresses_one_frame(extraction_meta):
    relevant_stress_field = (
        extraction_meta.frame.fieldOutputs["S"]
        .getSubset(position=abaqusConstants.CENTROID)
        .getSubset(region=extraction_meta.odb_instance)
    )

    for one_value in relevant_stress_field.values:
        yield ElementStress(
            frame_rowid=None,
            elem_num=one_value.elementLabel,
            SP1=one_value.minPrincipal,
            SP2=one_value.midPrincipal,
            SP3=one_value.maxPrincipal,
            von_mises=one_value.mises,
        )


def _get_element_fallback_field(extraction_meta, NT_type):
    """In case something which is meant to be there isn't in the results."""

    fallback_field = (
        extraction_meta.frame.fieldOutputs["S"]
        .getSubset(position=abaqusConstants.CENTROID)
        .getSubset(region=extraction_meta.odb_instance)
    )

    working_dict = {fn: 0.0 for fn in NT_type.odb_value_field_names}

    for one_value in fallback_field.values:
        working_dict["frame_rowid"] = None
        working_dict["elem_num"] = one_value.elementLabel

        yield NT_type(**working_dict)


def get_strain_results_PEEQ_one_frame(extraction_meta):

    try:
        relevant_peeq_field = (
            extraction_meta.frame.fieldOutputs["PEEQ"]
            .getSubset(position=abaqusConstants.CENTROID)
            .getSubset(region=extraction_meta.odb_instance)
        )

    except KeyError:
        # No PEEQ in Aba 2021
        for elem_peeq in _get_element_fallback_field(extraction_meta, ElementPEEQ):
            yield elem_peeq

        raise StopIteration()

    # If we got here there are good results.
    for one_value in relevant_peeq_field.values:
        yield ElementPEEQ(
            frame_rowid=None,
            elem_num=one_value.elementLabel,
            PEEQ=one_value.data,
        )


def get_strain_results_ESEDEN_one_frame(extraction_meta):
    relevant_peeq_field = (
        extraction_meta.frame.fieldOutputs["ESEDEN"]
        .getSubset(position=abaqusConstants.WHOLE_ELEMENT)
        .getSubset(region=extraction_meta.odb_instance)
    )

    for one_value in relevant_peeq_field.values:
        yield ElementEnergyElastic(
            frame_rowid=None,
            elem_num=one_value.elementLabel,
            ESEDEN=one_value.data,
        )


def get_strain_results_EPDDEN_one_frame(extraction_meta):
    relevant_peeq_field = (
        extraction_meta.frame.fieldOutputs["EPDDEN"]
        .getSubset(position=abaqusConstants.WHOLE_ELEMENT)
        .getSubset(region=extraction_meta.odb_instance)
    )

    for one_value in relevant_peeq_field.values:
        yield ElementEnergyPlastic(
            frame_rowid=None,
            elem_num=one_value.elementLabel,
            EPDDEN=one_value.data,
        )


def get_element_node_force_one_frame(extraction_meta):
    """Have to grab the results from a few places... annoying!"""

    def get_one_direction_values(field_key):
        relevant_nforce_field = (
            extraction_meta.frame.fieldOutputs[field_key]
            .getSubset(position=abaqusConstants.ELEMENT_NODAL)
            .getSubset(region=extraction_meta.odb_instance)
        )

        for one_value in relevant_nforce_field.values:
            yield one_value.elementLabel, one_value.nodeLabel, one_value.data

    elem_to_nodes_in_order = collections.defaultdict(list)

    for elem_id, node_id, _ in get_one_direction_values("NFORC1"):
        elem_to_nodes_in_order[elem_id].append(node_id)

    nforc1 = {
        (elem_id, node_id): value
        for elem_id, node_id, value in get_one_direction_values("NFORC1")
    }
    nforc2 = {
        (elem_id, node_id): value
        for elem_id, node_id, value in get_one_direction_values("NFORC2")
    }

    for elem_id, node_list in elem_to_nodes_in_order.items():
        nf1 = [nforc1[(elem_id, node_id)] for node_id in node_list]
        nf2 = [nforc2[(elem_id, node_id)] for node_id in node_list]
        one_elem_data = []
        for nfxy in itertools.izip_longest(nf1, nf2):
            one_elem_data.extend(nfxy)

        local_norm = math.sqrt(sum([x**2 for x in one_elem_data]))
        one_elem_data.append(local_norm)

        # Add the "frame" and "elem_num" bits.
        one_elem_data.insert(0, None)
        one_elem_data.insert(1, elem_id)
        yield ElementNodeForces(*one_elem_data)


class StressResultAggregator:
    working_min = None
    working_max = None

    def __init__(self):
        self.working_max = dict()
        self.working_min = dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update_frame_results(self, element_stress_iterable):
        for elem_stress in element_stress_iterable:
            elem_num = elem_stress.elem_num
            this_val = elem_stress.von_mises

            for min_max_dict, min_max_func in (
                (self.working_min, min),
                (self.working_max, max),
            ):
                if elem_num in min_max_dict:
                    new_val = min_max_func(min_max_dict[elem_num], this_val)

                else:
                    new_val = this_val

                min_max_dict[elem_num] = new_val

    def make_goodman_results(self):
        elems = set()
        elems.update(self.working_min.keys())
        elems.update(self.working_max.keys())

        for elem in sorted(elems):
            sigma_max = self.working_max[elem]
            sigma_min = self.working_min[elem]

            s_amp = 0.5 * (sigma_max - sigma_min)
            s_mean = 0.5 * (sigma_max + sigma_min)
            l_goodman = s_amp / TEMP_SIGMA_ENDURANCE + s_mean / TEMP_SIGMA_UTS

            yield ElementFatigueResult(
                frame_rowid=None,
                elem_num=elem,
                SAmp=s_amp,
                SMean=s_mean,
                LGoodman=l_goodman,
            )


def _add_with_zero_pad(a, b):
    """Add two numpy arrays together, zero padding as needed."""
    a_shape = a.shape
    b_shape = b.shape

    if len(a_shape) != 1 or len(b.shape) != 1:
        print_in_term("Can't add these guys!")
        raise ValueError(a_shape, b_shape)

    max_len = max(a_shape[0], b_shape[0])

    working = numpy.zeros(shape=(max_len,))
    working[: a_shape[0]] = a
    working[: b_shape[0]] += b
    return working


def get_node_position_one_frame(extraction_meta):
    relevant_disp_field = extraction_meta.frame.fieldOutputs["U"].getSubset(
        region=extraction_meta.odb_instance
    )

    for one_value in relevant_disp_field.values:
        overall_pos = _add_with_zero_pad(
            _get_data_array_as_double(one_value),
            extraction_meta.node_init_pos[one_value.nodeLabel],
        )

        yield NodePos(
            frame_rowid=None,
            node_num=one_value.nodeLabel,
            X=overall_pos[0],
            Y=overall_pos[1],
            Z=overall_pos[2],
        )


def get_node_reaction_one_frame(extraction_meta):
    relevant_disp_field = extraction_meta.frame.fieldOutputs["RF"].getSubset(
        region=extraction_meta.odb_instance
    )
    zero = numpy.array([0.0, 0.0, 0.0])
    for one_value in relevant_disp_field.values:
        if one_value.nodeLabel in extraction_meta.nodes_enforced_displacement:
            overall_pos = _add_with_zero_pad(_get_data_array_as_double(one_value), zero)
            yield NodeReact(
                frame_rowid=None,
                node_num=one_value.nodeLabel,
                X=overall_pos[0],
                Y=overall_pos[1],
                Z=overall_pos[2],
            )


def get_results_one_frame(extraction_meta):
    res_funcs = [
        get_stresses_one_frame,
        get_strain_results_PEEQ_one_frame,
        get_strain_results_ESEDEN_one_frame,
        get_strain_results_EPDDEN_one_frame,
        get_node_position_one_frame,
        get_node_reaction_one_frame,
        get_element_node_force_one_frame,
    ]
    for f in res_funcs:
        for row in f(extraction_meta):
            yield row


def extract_file_results(fn_odb):
    this_odb = odbAccess.openOdb(fn_odb)

    with Datastore(
        fn=db_fn
    ) as datastore, StressResultAggregator() as stress_results_aggregator:
        last_frame_db = None
        for frame_db, extraction_meta in walk_file_frames(
            EXTRACT_ALL_STEPS, this_odb, override_z_val
        ):
            print_in_term(frame_db)
            last_frame_db = frame_db
            all_results = get_results_one_frame(extraction_meta)

            datastore.add_frame_and_results(frame_db, all_results)

        # The goodman results need to be extracted from stresses at all frames on the last step.
        for frame_db, extraction_meta in walk_file_frames(
            True, this_odb, override_z_val
        ):

            is_release_oscillate_step = frame_db.step_num == STEP_IDX_RELEASE_OSCILLATE
            if is_release_oscillate_step:
                stress_result = get_stresses_one_frame(extraction_meta)
                stress_results_aggregator.update_frame_results(stress_result)

        datastore.add_results_on_existing_frame(
            last_frame_db, stress_results_aggregator.make_goodman_results()
        )

        datastore.add_many_history_results(get_history_results(this_odb))

    this_odb.close()


if __name__ == "__main__":
    extract_file_results(fn_odb)
