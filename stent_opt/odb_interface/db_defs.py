import collections

### Read by the code which parses .odbs

### Written by the code which parses .odbs


"""Main frames are referenced by the "frame_rowid" column on the other results"""

Frame = collections.namedtuple("Frame", (
    "rowid",
    "fn_odb",
    "part_name",
    "step_num",
    "step_name",
    "frame_id",
    "frame_value",
    "simulation_time",
))

make_frame = """CREATE TABLE IF NOT EXISTS Frame(
rowid INTEGER PRIMARY KEY,
fn_odb TEXT,
part_name TEXT,
step_num INTEGER,
step_name TEXT,
frame_id INTEGER,
frame_value REAL,
simulation_time REAL
);
"""


all_types_and_tables = [
    (Frame, make_frame),
]

def _make_nt_and_table_create(table_name, field_names):
    """This is a horrible global mutation sideffect-y thing but hopefully I'll
    never have to think about it again."""

    nt_fields = ["frame_rowid", "elem_num"]
    nt_fields.extend(field_names)
    NT_type = collections.namedtuple(table_name, nt_fields)

    create_table_fields = ["{0} REAL".format(f_name) for f_name in field_names]

    create_table_string = """CREATE TABLE IF NOT EXISTS {0}(
frame_rowid REFERENCES Frame(rowid), 
elem_num INTEGER,
{1}
)""".format(table_name, ", \n".join(create_table_fields))

    all_types_and_tables.append(
        (NT_type, create_table_string)
    )

    return NT_type, create_table_string


ElementStress, make_element_stress = _make_nt_and_table_create("ElementStress", ("SP1", "SP2", "SP3"))
ElementPEEQ, make_element_peeq = _make_nt_and_table_create("ElementPEEQ", ("PEEQ", ))




if __name__ == "__main__":
    ee = ElementStress("a", "b", 1, 2, 3)


