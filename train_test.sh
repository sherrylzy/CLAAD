#!/bin/bash

set -ex

CLAAD_PATH=./CLAAD

export PYTHONPATH=${CLAAD_PATH}:${PYTHONPATH}
python CLAAD/Source/trainer.py