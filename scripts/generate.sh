#! /bin/sh

export DOMAIN=$1
shift

if [ -z "$DOMAIN" ]
then
    echo "Usage: ./scripts/generate.sh <DOMAIN>"
    exit
fi

# Setup environment. 
export DOMAIN_PATH=./domains/$DOMAIN/
export DOMAIN_DATA_DIR=./data/inputs/$DOMAIN/
export PYTHONPATH=$(pwd)
. $DOMAIN_PATH/env.sh

# N.B.: assumes script is called from parent directory, as described in README.md
mkdir -p $DOMAIN_DATA_DIR
python3 -m pdb -c continue ./apps/generate.py --domain $DOMAIN $@
