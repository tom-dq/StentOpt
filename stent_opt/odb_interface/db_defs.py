import collections

### Read by the code which parses .odbs

### Written by the code which parses .odbs


"""Main frames are referenced by the "frame_rowid" column on the other results"""

Frame = collections.namedtuple("Frame", (
    "rowid",
    "fn_odb",
    "instance_name",
    "step_num",
    "step_name",
    "frame_id",
    "frame_value",
    "simulation_time",
))

make_frame = """CREATE TABLE IF NOT EXISTS Frame(
rowid INTEGER PRIMARY KEY,
fn_odb TEXT,
instance_name TEXT,
step_num INTEGER,
step_name TEXT,
frame_id INTEGER,
frame_value REAL,
simulation_time REAL
);
"""


HistoryResult = collections.namedtuple("HistoryResult", (
    "rowid",
    "fn_odb",
    "step_num",
    "step_name",
    "history_region",
    "history_identifier",
    "simulation_time",
    "history_value",
))

make_history = """CREATE TABLE IF NOT EXISTS HistoryResult(
rowid INTEGER PRIMARY KEY,
fn_odb TEXT,
step_num INTEGER,
step_name TEXT,
history_region TEXT,
history_identifier TEXT,
simulation_time REAL,
history_value REAL
);
"""



all_nt_and_table_defs = [
    (Frame, make_frame),
    (HistoryResult, make_history)
]

class ResultEntity:
    node_num = "node_num"
    elem_num = "elem_num"


def _make_nt_and_table_create(table_name, result_entity, field_names):
    """This is a horrible global mutation sideffect-y thing but hopefully I'll
    never have to think about it again."""

    nt_fields = ["frame_rowid", result_entity]
    nt_fields.extend(field_names)
    NT_type = collections.namedtuple(table_name, nt_fields)

    create_table_fields = ["{0} REAL".format(f_name) for f_name in field_names]

    create_table_string = """CREATE TABLE IF NOT EXISTS {0}(
frame_rowid REFERENCES Frame(rowid), 
{1} INTEGER,
{2}
)""".format(table_name, result_entity, ", \n".join(create_table_fields))

    all_nt_and_table_defs.append( (NT_type, create_table_string) )

    return NT_type


NodePos = _make_nt_and_table_create("NodePos", ResultEntity.node_num, ("X", "Y", "Z"))
ElementStress = _make_nt_and_table_create("ElementStress", ResultEntity.elem_num, ("SP1", "SP2", "SP3", "von_mises"))
ElementPEEQ = _make_nt_and_table_create("ElementPEEQ", ResultEntity.elem_num, ("PEEQ", ))
ElementEnergyElastic = _make_nt_and_table_create("ElementEnergyElastic", ResultEntity.elem_num, ("ESEDEN", ))
ElementEnergyPlastic = _make_nt_and_table_create("ElementEnergyPlastic", ResultEntity.elem_num, ("EPDDEN",))
ElementFatigueResult = _make_nt_and_table_create("ElementFatigueResult", ResultEntity.elem_num, ("SAmp", "SMean", "LGoodman",))

expected_history_results = ["ALLSE", "ALLPD", "ALLKE", "ALLWK"]


if __name__ == "__main__":
    ee = ElementStress("a", "b", 1, 2, 3)


