# This is run by Abaqus Python and populates the output odb.
# abaqus cae noGui=odb_extract.py -- c:\temp\db.db
# The double dash is for arguments passed to the script but ignored by Abaqus itself.

# Rule of thumb - make everything a double!

SAVE_IMAGES = True

import sys
import collections

import abaqusConstants
import odbAccess

if SAVE_IMAGES:
    import abaqus



LAST_FRAME_OF_STEP = True

from datastore import Datastore
from db_defs import Frame, NodePos, ElementStress, ElementPEEQ

# Get the command line option (should be last!).
fn_odb = sys.argv[-2]
db_fn = sys.argv[-1]


#fn_odb = r"C:\TEMP\aba\stent-36.odb"
#db_fn = r"C:\TEMP\aba\db-9.db"

ExtractionMeta = collections.namedtuple("ExtractionMeta", (
    "frame",
    "node_labels",
    "elem_labels",
    "node_init_pos",
    "odb_instance",
))

def print_in_term(x):
    # Old Python Syntax!
    print >> sys.__stdout__, x


def _get_nodes_on_elements(elements, ):
    all_nodes = set()
    for elem in elements:
        all_nodes.update(elem.connectivity)

    return frozenset(all_nodes)


def _get_node_initial_pos(instance):
    out_dict = {}
    for one_node in instance.nodes:
        out_dict[one_node.label] = one_node.coordinates.astype("d")

    return out_dict

def _get_data_array_as_double(one_value):
    if one_value.precision == abaqusConstants.SINGLE_PRECISION:
        return one_value.data.astype("d")

    elif one_value.precision == abaqusConstants.DOUBLE_PRECISION:
        return one_value.dataDouble

    else:
        raise TypeError(one_value.precision)


def walk_file_frames(fn_odb):
    """
    :return type: Iterable[Frame, ExtractionMeta]
    """
    this_odb = odbAccess.openOdb(fn_odb)

    for one_instance_name, instance in this_odb.rootAssembly.instances.items():
        this_part_elem_labels = frozenset(elem.label for elem in instance.elements)
        this_part_node_labels = _get_nodes_on_elements(instance.elements)
        this_part_node_initial_pos = _get_node_initial_pos(instance)
        previous_step_time = 0.0
        for step_num, (step_name, one_step) in enumerate(this_odb.steps.items()):

            last_frame_id_of_step = len(one_step.frames) - 1

            for frame_id, frame in enumerate(one_step.frames):
                should_yield = (not LAST_FRAME_OF_STEP or frame_id==last_frame_id_of_step)
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
                            simulation_time=previous_step_time+frame.frameValue,
                        ),
                        ExtractionMeta(
                            frame=frame,
                            node_labels=this_part_node_labels,
                            elem_labels=this_part_elem_labels,
                            node_init_pos=this_part_node_initial_pos,
                            odb_instance=instance,
                        )
                    )

                if frame_id == last_frame_id_of_step:
                    previous_step_time += frame.frameValue

                    if SAVE_IMAGES:
                        # Print to file?
                        fn = r"c:\temp\aba\Mod-{0}-{1}-{2}.png".format(one_instance_name, step_name, frame_id)
                        abaqus.session.printOptions.setValues(vpBackground=True)
                        abaqus.session.printToFile(fn, format=abaqusConstants.PNG)

    this_odb.close()


def get_stresses_one_frame(extraction_meta):
    relevant_stress_field = (
        extraction_meta
            .frame
            .fieldOutputs['S']
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


def get_strain_results_PEEQ_one_frame(extraction_meta):
    relevant_peeq_field = (
        extraction_meta
            .frame
            .fieldOutputs['PEEQ']
            .getSubset(position=abaqusConstants.CENTROID)
            .getSubset(region=extraction_meta.odb_instance)
    )

    for one_value in relevant_peeq_field.values:
        yield ElementPEEQ(
            frame_rowid=None,
            elem_num=one_value.elementLabel,
            PEEQ=one_value.data,
        )


def get_node_position_one_frame(extraction_meta):
    relevant_disp_field = (
        extraction_meta
            .frame
            .fieldOutputs['U']
            .getSubset(region=extraction_meta.odb_instance)
    )

    for one_value in relevant_disp_field.values:
        overall_pos = _get_data_array_as_double(one_value) + extraction_meta.node_init_pos[one_value.nodeLabel]
        yield NodePos(
            frame_rowid=None,
            node_num=one_value.nodeLabel,
            X=overall_pos[0],
            Y=overall_pos[1],
            Z=overall_pos[2]
        )


def get_results_one_frame(extraction_meta):
    res_funcs = [
        get_stresses_one_frame,
        get_strain_results_PEEQ_one_frame,
        get_node_position_one_frame,
    ]
    for f in res_funcs:
        for row in f(extraction_meta):
            yield row


def extract_file_results(fn_odb):
    with Datastore(fn=db_fn) as datastore:
        for frame_db, extraction_meta in walk_file_frames(fn_odb):
            print_in_term(frame_db)
            all_results = get_results_one_frame(extraction_meta)
            datastore.add_frame_and_results(frame_db, all_results)



if __name__ == "__main__":
    extract_file_results(fn_odb)


