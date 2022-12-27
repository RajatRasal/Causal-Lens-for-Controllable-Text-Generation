#!/bin/bash

MODEL=./lightning_logs/version_8/checkpoints/epoch=0-step=469000.ckpt

# poetry run python3 -m src.experiments.sentiment_classification \
#     -i $MODEL \
#     -o ./lightning_logs_classify_train/ \
#     -ds 1000 \
#     -lf 1

poetry run python3 -m src.experiments.sentiment_classification \
    -i $MODEL \
    -o ./lightning_logs_classify_train/ \
    -ds 100000 \
    -lf 50 \
    -b 256 \
    -s 100
