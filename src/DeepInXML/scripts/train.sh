#!/bin/sh
export PYTHONPATH=$(pwd)/

# Start training
python3 apps/train.py $@
