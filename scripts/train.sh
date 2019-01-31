#!/bin/sh
export PYTHONPATH=$(pwd)/

# Start training
python3 -m pdb -c continue apps/train.py $@
