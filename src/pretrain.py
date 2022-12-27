import random

import pytorch_lightning as pl
from datasets import load_dataset
from lightning_lite.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from .data import TokenisedSentencesFromIterable, collate_tokens
from .tokeniser import bert_pretrained_tokeniser, gpt2_pretrained_tokeniser
from .vae import BertGPT2VAE

# TODO: Output kl, loss, elbo etc. to Tensorboard
# TODO: Interpolations
# TODO: Currently we do not include an h_emb as seen here
# https://github.com/ChunyuanLI/Optimus/blob/f63f4a7ca10aea022978500a37d72dd53a37a576/code/pytorch_transformers/modeling_gpt2.py#L472
# We should still be able to get some good results since the paper
# says the h_mem is more important.

seed_everything(42, workers=True)

# Training dataset
tokeniser_encoder = bert_pretrained_tokeniser()
tokeniser_decoder = gpt2_pretrained_tokeniser()
train_list = load_dataset("wikitext", "wikitext-2-v1", cache_dir="./data")[
    "train"
]["text"]
train_dataset = TokenisedSentencesFromIterable(
    train_list, tokeniser_encoder, tokeniser_decoder
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=5,
    collate_fn=collate_tokens,
    num_workers=32,
)

# Max epochs and steps needed for beta scheduler
max_epochs = 1
max_steps = max_epochs * len(train_dataset.it)

# Validation dataset
# TODO: We need 5x as much data.
# Performance of LMs is linear in the log-scale of the no. of words.
# https://aclanthology.org/2021.acl-long.90.pdf
total_val_list = load_dataset("wikitext", "wikitext-2-v1", cache_dir="./data")[
    "train"
]["text"]
val_sample = random.sample(total_val_list, 10)
val_dataset = TokenisedSentencesFromIterable(
    val_sample, tokeniser_encoder, tokeniser_decoder
)
assert len(val_dataset.it) < 100
val_dataloader = DataLoader(
    val_dataset,
    batch_size=5,
    collate_fn=collate_tokens,
    num_workers=32,
)

# Defining the model
model = BertGPT2VAE(tokeniser_encoder, tokeniser_decoder)

# trainer = pl.Trainer(
#     max_epochs=40,
#     val_check_interval=100,
#     accelerator="gpu", devices="-1", strategy="ddp"
# )

log_and_val_freq = 1000
trainer = pl.Trainer(
    max_epochs=max_epochs,
    val_check_interval=log_and_val_freq,
    log_every_n_steps=log_and_val_freq,
    accelerator="gpu",
    devices=[0],
    max_steps=max_steps,
)

trainer.fit(model, train_dataloader, val_dataloader)
