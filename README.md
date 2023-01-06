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
1. `poetry update && poetry install`

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

## Available Pretrained Optimus Models
#### TODO: List of Optimus models available goes here

## Training
To kill any unstopped Python processes using GPUS:
```
nvidia-smi | grep 'py' | awk '{ print $5 }' | xargs -n1 kill -9
```

To start training:
```
poetry run python -m src.optimus.pretrain
```

To run tensorboard:
```
tensorboard --logdir lightning_logs/
```

## Experiments

### Guided Language Generation

**Sentence transfer via arithmetic** - Encode sentences $x_{A,B,C}$ to latent representations $z_{A,B,C}$. Then calculate $z_D = z_B − z_A + z_C$, and generate the sentence $x_D$ using the decoder. We can do this using the pre-trained Optimus and an Optimus fine-tuned on Yelp.

**Latent Interpolation** - Encode sentences $x_{1,2}$ to latent representations $z_{1,2}$. Then interpolate between the latent features by calculating $z_{\tau} = z_1 · (1 − \tau) + z_2 · \tau$, and generating the corresponding sentence $x_\tau$ using the decoder. We can do this using the pre-trained Optimus and an Optimus fine-tuned on Yelp.

**Label-conditional text generation** - The goal is to generate text reviews given the positive/negative sentiment. We fine-tune OPTIMUS using the VAE objective on the Yelp reviews polarity dataset, then freeze backbone weights. A conditional GAN is trained on the fixed latent space. The generation process is to first produce a latent vector zy based on a given label y using conditional GAN, then generate sentences conditioned on zy using the decoder.

1. Fine-tune a pretrained model on the Yelp dataset.
1. Use ARAE to fit a conditional GAN to the fine-tuned latent space.
1. Generate arbitrary sentences and sentences by style-transfer using the conditional decoder.

### Low-Resource Language Understanding

The output from the penultimate layer of the BERT Encoder, $h_{\text{[cls]}}$, is fed into an linear classifier $W_C \in \mathbb{R}^{K \times H}$, where $K$ is the number of classes, with objective $-\log(\text{softmax}(h_{\text{[cls]}}W^T_C))$.

Two schemes are used:
- Fine-tuning - where both the pre-trained model and the classifier are updated.
- Feature-based - where pre-trained model weights are frozen to provide embeddings for the classifier update.

**Sentiment classification** - A varying number of training samples are randomly
chosen, ranging from 1 to 10K per class, from the Yelp reviews polarity dataset. 10 trials are used when the number of available training samples are small, each is trained in 100 training epochs.

**Visualization of the latent space** - We use t-SNE to visualize the latent features $z$ on a 2D map, prior to fine-tuning for the sentiment classification problem. The validation set of Yelp is used to extract the latent features.

## Helpful Links
- [Methods of decoding tokens](https://huggingface.co/blog/how-to-generate)
- [Picking the right GPU for training](https://towardsdatascience.com/choosing-the-right-gpu-for-deep-learning-on-aws-d69c157d8c86)
