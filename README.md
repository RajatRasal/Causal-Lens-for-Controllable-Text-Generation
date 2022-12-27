# Causal-Lens-for-Controllable-Text-Generation
An implementation of the paper "Causal Lens for Controllable Text Generation" by Hu and Li 2021.

## Setup

1. Add the following to the local `.git/config`:
```
[user]
    name = Rajat Rasal
    email = yugiohrajat1@gmail.com
[core]
    editor = vim
```
1. `conda env update -f environment.yml`
1. `conda activate causal_control_text_gen`
1. `pre-commit install`
1. `poetry install`

#### Note: Tested on Mac OS and Amazon P3.8xlarge with Deep Learning AMI.
```
python3 -c "import torch; print(torch.cuda.device_count())"
```

## Data

The Wikipedia-2 dataset, from GPT-2 paper, has been pre-processed such that each sentence has a maximum length of 64 or the tokenized sentence length is smaller than 256.

```
wget -O data/wikipedia_json_64_filtered.zip https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/wikipedia_json_64_filtered.zip
```

The full dataset can also be downloaded.
```
wget -O data/wikipedia.segmented.nltk.txt https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/wikipedia.segmented.nltk.txt
```

## Training
To kill any unstopped Python processes using GPUS:
```
nvidia-smi | grep 'py' | awk '{ print $5 }' | xargs -n1 kill -9
```

To start training:
```
poetry run python -m src.pretrain
```

To run tensorboard:
```
tensorboard --logdir lightning_logs/
```


## Helpful Links
- [Methods of decoding tokens](https://huggingface.co/blog/how-to-generate)
- [Picking the right GPU for training](https://towardsdatascience.com/choosing-the-right-gpu-for-deep-learning-on-aws-d69c157d8c86)
