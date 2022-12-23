from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import (  # noqa: E501
    BertModel,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


class VAE(pl.LightningModule):
    def __init__(self, latent_size: int = 32):
        super().__init__()
        self.save_hyperparameters()

        # Encoder Tokeniser
        self.tokeniser_encoder = BertTokenizer.from_pretrained(
            "bert-base-cased"
        )

        # Encoder Model
        self.model_encoder = BertModel.from_pretrained("bert-base-cased")

        # Decoder Tokeniser
        self.tokeniser_decoder = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokeniser_decoder.add_special_tokens(
            {"pad_token": "<PAD>", "bos_token": "<BOS>", "eos_token": "<EOS>"}
        )

        # Decoder Model
        self.model_decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model_decoder.resize_token_embeddings(len(self.tokeniser_decoder))

        # Bottleneck
        self.latent_proj = nn.Linear(
            self.model_encoder.config.hidden_size,
            self.hparams.latent_size * 2,
            bias=False,
        )

        # Decoder memory embedding projection, different latent vector for
        # each layer
        self.memory_emb_flat = nn.Linear(
            self.hparams.latent_size,
            self.model_decoder.config.n_embd
            * self.model_decoder.config.n_layer,
            bias=False,
        )

    def _reparametrise(self, mean: torch.Tensor, logvar: torch.Tensor):
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, sentences: List[str]):
        # Tokenise for encoder and decoder
        enc_tokens = self.tokeniser_encoder(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        dec_tokens = self.tokeniser_decoder(
            sentences,
            padding=True,
            return_tensors="pt",
        )

        # Encoding
        pooled_encoder_output = self.model_encoder(
            **enc_tokens, output_hidden_states=True
        ).pooler_output

        # Bottleneck
        mean, logvar = self.latent_proj(pooled_encoder_output).chunk(2, -1)
        latent = self._reparametrise(mean, logvar)

        # [batch_size, n_embd * n_layer]
        memory_latent = self.memory_emb_flat(latent)
        memory_latent_per_layer = torch.split(
            memory_latent.unsqueeze(1), self.model_decoder.config.n_embd, dim=2
        )

        # [batch_size, num_heads, seq_length = 1, head_dim]
        m = [_latent.view(3, 12, 1, 64) for _latent in memory_latent_per_layer]
        m = tuple(zip(m, m))

        # Decoding
        # print(dec_tokens["attention_mask"])
        # outputs = self.model_decoder(**dec_tokens, past_key_values=xx,
        # labels=dec_tokens["input_ids"], return_dict=True)
        outputs = self.model_decoder(
            dec_tokens["input_ids"],
            past_key_values=m,
            labels=dec_tokens["input_ids"],
            return_dict=True,
        )
        print(outputs["loss"])

    def training_step(self, batch, batch_idx):
        self.forward()
        pass


if __name__ == "__main__":
    model = VAE()
    sentences = ["xxx", "yyy", "my name is rajat"]
    model(sentences)
