# This is run by Abaqus Python and populates the output odb.
# abaqus cae noGui=odb_extract.py -- c:\temp\db.db
# The double dash is for arguments passed to the script but ignored by Abaqus itself.


import sys
import collections

import abaqusConstants
import odbAccess

fn_odb = r"C:\TEMP\aba\stent-13.odb"

from datastore import Datastore
from db_defs import Frame, ElementStress

# Get the command line option (should be last!).
db_fn = sys.argv[-1]

ExtractionMeta = collections.namedtuple("ExtractionMeta", (
    "frame",
    "node_labels",
    "elem_labels",
))

def print_in_term(x):
    # Old Python Syntax!
    print >> sys.__stdout__, x


def _get_nodes_on_elements(elements, ):
    all_nodes = set()
    for elem in elements:
        all_nodes.update(elem.connectivity)

    return frozenset(all_nodes)


def walk_file_frames(fn_odb):
    """
    :return type: Iterable[Frame, ExtractionMeta]
    """
    this_odb = odbAccess.openOdb(fn_odb)
    print_in_term(this_odb)

    for one_part_name, part in this_odb.rootAssembly.instances.items():
        print_in_term(one_part_name)
        this_part_elem_labels = frozenset(elem.label for elem in part.elements)
        this_part_node_labels = _get_nodes_on_elements(part.elements)
        previous_step_time = 0.0
        for step_num, (step_name, one_step) in enumerate(this_odb.steps.items()):
            print_in_term(step_name)

            last_frame_id_of_step = len(one_step.frames) - 1

            for frame_id, frame in enumerate(one_step.frames):
                yield (
                    Frame(
                        rowid=None,
                        fn_odb=fn_odb,
                        part_name=one_part_name,
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
                    )
                )

                if frame_id == last_frame_id_of_step:
                    previous_step_time += frame.frameValue


    this_odb.close()


def get_stresses_one_frame(extraction_meta):
    frame = extraction_meta.frame
    stress_field_integration_points = frame.fieldOutputs['S']
    stress_field_centroid = stress_field_integration_points.getSubset(position=abaqusConstants.CENTROID)

    for one_value in stress_field_centroid.values:
        if one_value.elementLabel in extraction_meta.elem_labels:
            yield ElementStress(
                frame_rowid=None,
                elem_num=one_value.elementLabel,
                SP1=one_value.minPrincipal,
                SP2=one_value.midPrincipal,
                SP3=one_value.maxPrincipal,
            )


def get_strain_results_PEEQ_one_frame(extraction_meta):
    frame = extraction_meta.frame
    stress_field_integration_points = frame.fieldOutputs['PEEQ']
    stress_field_centroid = stress_field_integration_points.getSubset(position=abaqusConstants.CENTROID)

    for one_value in stress_field_centroid.values:
        if one_value.elementLabel in extraction_meta.elem_labels:
            yield ElementStress(
                frame_rowid=None,
                elem_num=one_value.elementLabel,
                PEEQ=one_value.data[0],
            )


def get_results_one_frame(extraction_meta):
    res_funcs = [
        get_stresses_one_frame,
        get_strain_results_PEEQ_one_frame,
    ]
    for f in res_funcs:
        for row in f(extraction_meta):
            yield row


def extract_file_results(fn_odb):
    with Datastore(fn=db_fn) as datastore:
        for frame_db, extraction_meta in walk_file_frames(fn_odb):
            print_in_term(frame_db)
            stress_results = get_stresses_one_frame(extraction_meta)
            datastore.add_frame_and_results(frame_db, stress_results)



if __name__ == "__main__":
    extract_file_results(fn_odb)


