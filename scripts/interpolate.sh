#!/bin/bash

MODEL=./lightning_logs/version_8/checkpoints/epoch=0-step=400000.ckpt

poetry run python \
    -m src.experiments.guided_language_generation \
    -f $MODEL \
    --mode interpolate \
    -a "children are looking for the water to be clear." \
    -b "there are two people playing soccer."
