from typing import List

import pytorch_lightning as pl
from torch import nn
from transformers import (  # noqa: E501
    BertModel,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


class VAE(pl.LightningModule):
    def __init__(self):
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
        self.encoder_linear = nn.Linear(
            self.model_encoder.config.hidden_size,
            self.model_encoder.config.hidden_size,
        )

    def forward(self, sentences: List[str]):
        tokens = self.tokeniser_encoder(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        hidden_states = self.model_encoder(**tokens, output_hidden_states=True)
        res = self.encoder_linear(hidden_states.pooler_output)
        print(res.shape)


if __name__ == "__main__":
    model = VAE()
    sentences = ["xxx", "yyy", "my name is rajat"]
    model(sentences)
