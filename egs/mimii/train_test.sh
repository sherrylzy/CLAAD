#!/bin/bash

set -ex

. ./path.sh

# python $CLAAD_ROOT_PATH/CLAAD/Source/trainer.py \
#     --config configs/example.yaml \
python $CLAAD_ROOT_PATH/CLAAD/bin/train.py \
    --config configs/example.yaml
