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



ElementStress = collections.namedtuple("ElementStress", (
    "frame_rowid",
    "elem_num",
    "SP1",
    "SP2",
    "SP3",
))

make_element_stress = """CREATE TABLE IF NOT EXISTS ElementStress(
frame_rowid REFERENCES Frame(rowid), 
elem_num INTEGER,
SP1 REAL,
SP2 REAL,
SP3 REAL
)"""


ElementStrain = collections.namedtuple("ElementStrainPEEQ", (
    "frame_rowid",
    "elem_num",
    "PEEQ",
))

make_element_strain = """CREATE TABLE IF NOT EXISTS ElementStrain(
frame_rowid REFERENCES Frame(rowid), 
elem_num INTEGER,
PEEQ REAL
)"""

all_types_and_tables = [
    (Frame, make_frame),
    (ElementStress, make_element_stress),
    (ElementStrain, make_element_strain),
]

if __name__ == "__main__":
    ee = ElementStress("a", "b", 1, 2, 3)


