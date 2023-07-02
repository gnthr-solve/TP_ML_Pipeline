#!/bin/bash

ML_Pipeline=$(grep 'name:' environment.yml | cut -d':' -f2 | xargs)
python -m venv $ML_Pipeline
source $ML_Pipeline/bin/activate
pip install -r requirements.txt
