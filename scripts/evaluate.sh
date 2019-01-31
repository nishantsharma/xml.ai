#!/bin/sh

export PYTHONPATH=$(pwd)/
# Start evaluation 
# python3 -m pdb -c continue ./apps/evaluate.py $@ 
python3 ./apps/evaluate.py $@ 2>&1 | less 
