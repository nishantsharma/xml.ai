#!/bin/sh

export PYTHONPATH=$(pwd)/
# Start evaluation 
python3 -m pdb -c continue ./apps/evaluate.py $@ 
