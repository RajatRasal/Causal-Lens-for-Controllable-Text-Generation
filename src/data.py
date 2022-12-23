from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from transformers import BertTokenizer, GPT2Tokenizer, PreTrainedTokenizer


@dataclass
class Tokens:
    enc_tokens: torch.Tensor
    dec_tokens: torch.Tensor
    sentence: torch.Tensor


class TokenisedSentences(IterableDataset):

    TOKENISER_ARGS = {
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
        "max_length": 512,
    }

    def __init__(
        self,
        file: str,
        tokeniser_encoder: PreTrainedTokenizer,
        tokeniser_decoder: PreTrainedTokenizer,
    ):
        self.file = file
        self.tokeniser_encoder = tokeniser_encoder
        self.tokeniser_decoder = tokeniser_decoder

    def __iter__(self) -> Iterator[Tokens]:
        with open(self.file) as f:
            while line := f.readline().replace("\n", ""):
                yield Tokens(
                    enc_tokens=self.tokeniser_encoder.encode(
                        text=line, **self.TOKENISER_ARGS
                    ),
                    dec_tokens=self.tokeniser_decoder.encode(
                        text=line, **self.TOKENISER_ARGS
                    ),
                    sentence=line,
                )


if __name__ == "__main__":
    # Encoder Tokeniser
    tokeniser_encoder = BertTokenizer.from_pretrained("bert-base-cased")
    # Decoder Tokeniser
    tokeniser_decoder = GPT2Tokenizer.from_pretrained("gpt2")
    tokeniser_decoder.add_special_tokens(
        {"pad_token": "<PAD>", "bos_token": "<BOS>", "eos_token": "<EOS>"}
    )

    file = "./data/wikipedia.segmented.nltk.txt"
    dataset = TokenisedSentences(file, tokeniser_encoder, tokeniser_decoder)

    for tokens, _ in zip(dataset, range(10)):
        print(tokens.enc_tokens.shape)
        print(tokens.dec_tokens.shape)
        print(len(tokens.sentence.split(" ")))
