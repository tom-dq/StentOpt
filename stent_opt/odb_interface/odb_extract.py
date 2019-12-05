# This is run by Abaqus Python and populates the output odb.
# abaqus cae noGui=odb_extract.py c:\temp\db.db

import sys

import abaqusConstants
import odbAccess

from .datastore import Datastore
from .db_defs import Frame, ElementStress

# Get the command line option.
db_fn = sys.argv[1]

def walk_file(fn_odb):
    this_odb =