export PYTHONPATH=$(pwd)/
INPUT_PATH=../../data/01_toy_reverse/
# Start training
python3 examples/sample.py --input_path $INPUT_PATH $*
