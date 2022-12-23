from dataclasses import dataclass
from typing import Iterator, List

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

from .tokeniser import bert_pretrained_tokeniser, gpt2_pretrained_tokeniser


@dataclass
class Tokens:
    enc_tokens: torch.Tensor
    dec_tokens: torch.Tensor
    sentence: torch.Tensor


@dataclass
class TokensBatch:
    enc_tokens_batch: torch.Tensor
    dec_tokens_batch: torch.Tensor
    sentences: List[str]


def collate_tokens(tokens_list: List[Tokens]) -> TokensBatch:
    enc_tokens_list = []
    dec_tokens_list = []
    sentences = []

    for tokens in tokens_list:
        enc_tokens_list.append(tokens.enc_tokens)
        dec_tokens_list.append(tokens.dec_tokens)
        sentences.append(tokens.sentence)

    return TokensBatch(
        enc_tokens_batch=torch.stack(enc_tokens_list).squeeze(1),
        dec_tokens_batch=torch.stack(dec_tokens_list).squeeze(1),
        sentences=sentences,
    )


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

    def build_tokens(self, line: str) -> Tokens:
        return Tokens(
            enc_tokens=self.tokeniser_encoder.encode(
                text=line, **self.TOKENISER_ARGS
            ),
            dec_tokens=self.tokeniser_decoder.encode(
                text=line, **self.TOKENISER_ARGS
            ),
            sentence=line,
        )

    def __iter__(self) -> Iterator[Tokens]:
        with open(self.file) as f:
            while line := f.readline().replace("\n", ""):
                yield self.build_tokens(line)


if __name__ == "__main__":
    tokeniser_encoder = bert_pretrained_tokeniser()
    tokeniser_decoder = gpt2_pretrained_tokeniser()

    file = "./data/wikipedia.segmented.nltk.txt"
    dataset = TokenisedSentences(file, tokeniser_encoder, tokeniser_decoder)

    dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate_tokens)

    for batch, _ in zip(dataloader, range(10)):
        print(batch.enc_tokens_batch.shape)
        print(batch.dec_tokens_batch.shape)
        print(len(batch.sentences))
