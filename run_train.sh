#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"

torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py "$@"