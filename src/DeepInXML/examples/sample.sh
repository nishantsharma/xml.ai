export PYTHONPATH=$(pwd)/
INPUT_PATH=../../data/toy_reverse/
# Start training
python3 examples/sample.py --input_path $INPUT_PATH $*
