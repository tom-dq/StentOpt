import collections

### Read by the code which parses .odbs

### Written by the code which parses .odbs


class IndexCreateStage:
    """Make these step by step so the inserts are faster"""

    primary_extract = 0
    composite = 1
    patch = 2


"""Main frames are referenced by the "frame_rowid" column on the other results"""

Frame = collections.namedtuple(
    "Frame",
    (
        "rowid",
        "fn_odb",
        "instance_name",
        "step_num",
        "step_name",
        "frame_id",
        "frame_value",
        "simulation_time",
    ),
)

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


HistoryResult = collections.namedtuple(
    "HistoryResult",
    (
        "rowid",
        "fn_odb",
        "step_num",
        "step_name",
        "history_region",
        "history_identifier",
        "simulation_time",
        "history_value",
    ),
)

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


all_nt_and_table_defs = [(Frame, make_frame), (HistoryResult, make_history)]

all_index_defs = []


class ResultEntity:
    node_num = "node_num"
    elem_num = "elem_num"


def make_index_name(NT_type):
    return "idx_{0}".format(NT_type.__name__)


def _make_nt_and_table_create(
    table_name, result_entity, field_names, index_create_stage
):
    """This is a horrible global mutation sideffect-y thing but hopefully I'll
    never have to think about it again."""

    nt_fields = ["frame_rowid", result_entity]
    nt_fields.extend(field_names)
    NT_type = collections.namedtuple(table_name, nt_fields)

    # Add a "get last field" method... so hacky!
    def get_last_value(self):
        return getattr(self, field_names[-1])

    NT_type.get_last_value = get_last_value

    # The names of all the fields which are "odb value data" - for where that needs to be populated with fallback.
    NT_type.odb_value_field_names = field_names

    create_table_fields = ["{0} REAL".format(f_name) for f_name in field_names]

    create_table_string = """CREATE TABLE IF NOT EXISTS {0}(
frame_rowid REFERENCES Frame(rowid), 
{1} INTEGER,
{2}
)""".format(
        table_name, result_entity, ", \n".join(create_table_fields)
    )

    all_nt_and_table_defs.append((NT_type, create_table_string))

    # Make an index for effective lookups
    # e.g. - "CREATE INDEX IF NOT EXISTS idx_AAA ON ElementEnergyElastic (frame_rowid, elem_num);"
    create_index = "CREATE INDEX IF NOT EXISTS {0} ON {1} (frame_rowid, {2})".format(
        make_index_name(NT_type), table_name, result_entity
    )
    drop_index = "DROP INDEX IF EXISTS {0}".format(make_index_name(NT_type))
    all_index_defs.append((index_create_stage, create_index, drop_index))

    return NT_type


# fmt:off
NodePos = _make_nt_and_table_create("NodePos", ResultEntity.node_num, ("X", "Y", "Z"), IndexCreateStage.primary_extract)
NodeReact = _make_nt_and_table_create("NodeReact", ResultEntity.node_num, ("X", "Y", "Z"), IndexCreateStage.primary_extract)
ElementStress = _make_nt_and_table_create("ElementStress", ResultEntity.elem_num, ("SP1", "SP2", "SP3", "von_mises"), IndexCreateStage.primary_extract)
ElementPEEQ = _make_nt_and_table_create("ElementPEEQ", ResultEntity.elem_num, ("PEEQ", ), IndexCreateStage.primary_extract)
ElementEnergyElastic = _make_nt_and_table_create("ElementEnergyElastic", ResultEntity.elem_num, ("ESEDEN", ), IndexCreateStage.primary_extract)
ElementEnergyPlastic = _make_nt_and_table_create("ElementEnergyPlastic", ResultEntity.elem_num, ("EPDDEN",), IndexCreateStage.primary_extract)
ElementFatigueResult = _make_nt_and_table_create("ElementFatigueResult", ResultEntity.elem_num, ("SAmp", "SMean", "LGoodman",), IndexCreateStage.primary_extract)
ElementNodeForces = _make_nt_and_table_create("ElementNodeForces", ResultEntity.elem_num, ("N1X", "N1Y", "N2X", "N2Y", "N3X", "N3Y", "N4X", "N4Y", "overall_norm"), IndexCreateStage.primary_extract)
ElementGlobalPatchSensitivity = _make_nt_and_table_create("ElementGlobalPatchSensitivity", ResultEntity.elem_num, ("gradient_from_patch",), IndexCreateStage.patch)
ElementCustomCompositeOne = _make_nt_and_table_create("ElementCustomCompositeOne", ResultEntity.elem_num, ("comp_val",), IndexCreateStage.composite)
ElementCustomCompositeTwo = _make_nt_and_table_create("ElementCustomCompositeTwo", ResultEntity.elem_num, ("comp_val",), IndexCreateStage.composite)
# fmt:on

expected_history_results = ["ALLSE", "ALLPD", "ALLKE", "ALLWK"]


if __name__ == "__main__":
    ee = ElementStress("a", "b", 1, 2, 3)
