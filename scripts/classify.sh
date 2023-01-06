#!/bin/bash

poetry run python3 -m src.experiments.sentiment_classification.train \
    --train-dataset-size 100000 \
    --val-dataset-size 1000 \
    --log-freq 50 \
    --batch-size 256 \
    --train-prop 0.9
