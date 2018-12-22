#!/bin/sh

export PYTHONPATH=$(pwd)/
# Start evaluation 
python3 ./apps/evaluate.py $@ 
