PYTHONPATH=$(pwd)/
TRAIN_PATH=../../data/01_toy_reverse/train/
DEV_PATH=../../data/01_toy_reverse/dev/
# Start training
python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH $*
