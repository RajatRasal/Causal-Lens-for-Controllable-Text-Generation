#!/bin/bash

MODEL=./lightning_logs/version_8/checkpoints/epoch=0-step=404000.ckpt

poetry run python \
    -m src.experiments.guided_language_generation \
    -f $MODEL \
    --mode arithmetic \
    -a "a girl makes a silly face" \
    -b "two soccer players are playing soccer" \
    -c "a girl poses for a picture"

poetry run python \
    -m src.experiments.guided_language_generation \
    -f $MODEL \
    --mode arithmetic \
    -a "a girl makes a silly face" \
    -b "two soccer players are playing soccer" \
    -c "a woman in a red scarf takes pictures of the stars"

poetry run python \
    -m src.experiments.guided_language_generation \
    -f $MODEL \
    --mode arithmetic \
    -a "a girl makes a silly face" \
    -b "two soccer players are playing soccer" \
    -c "a girl in a blue dress is taking pictures of a microscope"

poetry run python \
    -m src.experiments.guided_language_generation \
    -f $MODEL \
    --mode arithmetic \
    -a "a girl makes a silly face" \
    -b "two soccer players are playing soccer" \
    -c "a boy is taking a bath"

poetry run python \
    -m src.experiments.guided_language_generation \
    -f $MODEL \
    --mode arithmetic \
    -a "a girl makes a silly face" \
    -b "two soccer players are playing soccer" \
    -c "a boy is taking a bath"
