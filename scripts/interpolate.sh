#!/bin/bash

poetry run python -m src.experiments.latent_arithmetic.interpolation \
    --sent-source "children are looking for the water to be clear." \
    --sent-target "there are two people playing soccer." \
    --seed 100

echo

# poetry run python -m src.experiments.latent_arithmetic.interpolation \
#     --sent-source "a woman is riding a moped on a street with large trees and other riders ride over it as some sort of bob lights are passing behind her." \
#     --sent-target "two men in blue holding each other standing the window of a one wheel bicycle trying out the tube."
