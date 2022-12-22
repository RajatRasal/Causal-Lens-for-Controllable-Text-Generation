# Causal-Lens-for-Controllable-Text-Generation
An implementation of the paper "Causal Lens for Controllable Text Generation" by Hu and Li 2021.

## Setup

1. Add the following to the local `.git/config`:
```
[user]
    name = Rajat Rasal
    email = yugiohrajat1@gmail.com
```
1. `conda env update -f environment.yml`
1. `conda activate causal_control_text_gen`
1. `pre-commit install`
1. `poetry install`

## Data

The Wikipedia-2 dataset, from GPT-2 paper, has been pre-processed such that each sentence has a maximum length of 64 or the tokenized sentence length is smaller than 256.

```
https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/wikipedia_json_64_filtered.zip
```
