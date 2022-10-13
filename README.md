# Topology Optimisation with Major Nonlinearities

### Note that this has not yet achieved what I wanted it to, and I don't have the time to continue pursuing it right at the moment. This repo is public so I can discuss this work, not necessarily because I think it would be a good use of your time, fair reader, to get it running and work with it.

## Overview

Main entry point is `stent_opt/make_stent.py`. If you want to get it up and running there may be a bit of messing with `PYTHONPATH` and the regular system `PATH` to get everything running.


### abaqus_model

The module `stent_opt/abaqus_model` contains Python objects to define an Abaqus input file. Note that this is a standalone module from Abaqus; it doesn't use the built-in Abaqus scripting interface and runs in Python 3. This outputs a standalone `.inp` file.

### odb_interface

`stent_opt/odb_interface` contains stuff which needs to be runnable in both the old Abaqus python (2.6 or 2.7), and whatever the most up to date Python is. The philosophy is "Get the data out we need out of the `.odb` files and old Python, into a sqlite database. From there we can continue with modern tooling."

- `db_defs.py` contain the types used to store data, and the function which creates them, `_make_nt_and_table_create`, makes the `CREATE TABLE...` statements while it's at it. 
- `datastore.py` is the interface to a sqlite database.
- `odb_extract.py` is what's run by the Abaqus built in python, e.g., `abaqus cae noGui=odb_extract.py -- c:\temp\db.db`

### struct_opt
`stent_opt/struct_opt` has the code for the actual optimisation, i.e., turn this element off or on for the next iteration.


