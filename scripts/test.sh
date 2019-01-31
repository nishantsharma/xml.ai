#!/bin/sh
export PYTHONPATH=$(pwd)/

TRAIN_PATH=data/toy_reverse/train/data.txt
DEV_PATH=data/toy_reverse/dev/data.txt

# Start training
python3 -m pdb -c continue ./apps/test.py

# Resume training
#python ./apps/test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --resume

# Load checkpoint
#python ./apps/test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --load_checkpoint $(ls -t experiment/checkpoints/ | head -1)
