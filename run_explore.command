#! /bin/bash
source venv-py39/bin/activate
export PYTHONPATH=$(pwd)
cd stent_opt/struct_opt
python explore.py
