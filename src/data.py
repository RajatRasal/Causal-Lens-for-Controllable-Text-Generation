from dataclasses import dataclass
from typing import Iterator, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

from .tokeniser import bert_pretrained_tokeniser, gpt2_pretrained_tokeniser


@dataclass
class Tokens:
    enc_tokens: torch.Tensor
    enc_tokens_length: int
    dec_tokens: torch.Tensor
    dec_tokens_length: int
    sentence: torch.Tensor


@dataclass
class TokensBatch:
    enc_tokens_batch: torch.Tensor
    enc_tokens_batch_lengths: List[int]
    dec_tokens_batch: torch.Tensor
    dec_tokens_batch_lengths: List[int]
    sentences: List[str]


def collate_tokens(tokens_list: List[Tokens]) -> TokensBatch:
    enc_tokens_list = []
    enc_tokens_lengths = []
    dec_tokens_list = []
    dec_tokens_lengths = []
    sentences = []

    for tokens in tokens_list:
        enc_tokens_list.append(tokens.enc_tokens)
        enc_tokens_lengths.append(tokens.enc_tokens_length)
        dec_tokens_list.append(tokens.dec_tokens)
        dec_tokens_lengths.append(tokens.dec_tokens_length)
        sentences.append(tokens.sentence)

    return TokensBatch(
        enc_tokens_batch=torch.stack(enc_tokens_list).squeeze(1),
        enc_tokens_batch_lengths=enc_tokens_lengths,
        dec_tokens_batch=torch.stack(dec_tokens_list).squeeze(1),
        dec_tokens_batch_lengths=dec_tokens_lengths,
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
        tokeniser_encoder: PreTrainedTokenizer,
        tokeniser_decoder: PreTrainedTokenizer,
    ):
        self.tokeniser_encoder = tokeniser_encoder
        self.tokeniser_decoder = tokeniser_decoder

    def build_tokens(self, line: str) -> Tokens:
        # TODO: Create our own Tokeniser wrapper which can do the
        # necesary preprocessing!!
        dec_sentence = (
            self.tokeniser_decoder.bos_token
            + line
            + self.tokeniser_decoder.eos_token
        )
        enc_tokens = self.tokeniser_encoder.encode(
            text=line, **self.TOKENISER_ARGS
        )
        enc_tokens_length = (
            (enc_tokens != self.tokeniser_encoder.pad_token_id).sum().item()
        )
        dec_tokens = self.tokeniser_decoder.encode(
            text=dec_sentence, **self.TOKENISER_ARGS
        )
        dec_tokens_length = (
            (dec_tokens != self.tokeniser_decoder.pad_token_id).sum().item()
        )
        return Tokens(
            enc_tokens=enc_tokens,
            enc_tokens_length=enc_tokens_length,
            dec_tokens=dec_tokens,
            dec_tokens_length=dec_tokens_length,
            sentence=line,
        )


class TokenisedSentencesFromFile(TokenisedSentences):
    def __init__(
        self,
        file: str,
        tokeniser_encoder: PreTrainedTokenizer,
        tokeniser_decoder: PreTrainedTokenizer,
    ):
        super().__init__(tokeniser_encoder, tokeniser_decoder)
        self.file = file

    def __iter__(self) -> Iterator[Tokens]:
        with open(self.file) as f:
            while line := f.readline():
                line = line.replace("\n", "")
                if not line:
                    continue
                yield self.build_tokens(line)


class TokenisedSentencesFromIterable(TokenisedSentences):
    def __init__(
        self,
        iterable,
        tokeniser_encoder: PreTrainedTokenizer,
        tokeniser_decoder: PreTrainedTokenizer,
    ):
        super().__init__(tokeniser_encoder, tokeniser_decoder)
        self.it = iterable

    def __iter__(self) -> Iterator[Tokens]:
        for line in self.it:
            # Some lines are blank or contain titles such as
            #  = = Cricket = =
            # which can be picked up with the startswith.
            if not line or line.startswith(" = "):
                continue
            # Split sentences
            # line[:-1] == "\n"
            # line[:-2] == " "
            sents = line[:-2].split(".")
            for sent in sents:
                if not sent:
                    continue
                _sent = sent + "."
                tokens = self.build_tokens(_sent)
                if tokens.enc_tokens_length <= 64:
                    yield tokens


if __name__ == "__main__":
    tokeniser_encoder = bert_pretrained_tokeniser()
    tokeniser_decoder = gpt2_pretrained_tokeniser()

    read_text_file = False

    if read_text_file:
        f = "./data/wikipedia.segmented.nltk.txt"
        ds = TokenisedSentences(f, tokeniser_encoder, tokeniser_decoder)

        # TODO: Include a bos, eos and pad token for GPT2 tokens
        dataloader = DataLoader(
            ds, batch_size=10000, collate_fn=collate_tokens, num_workers=10
        )

        for i, x in enumerate(dataloader):
            print(x.enc_tokens_batch[0])
            break
            if i % 100 == 0:
                print(i)
    else:
        iterable = load_dataset(
            "wikitext", "wikitext-2-v1", cache_dir="./data"
        )["train"]["text"]
        print(len(iterable))
        dataset = TokenisedSentencesFromIterable(
            iterable, tokeniser_encoder, tokeniser_decoder
        )
        dataloader = DataLoader(
            dataset, batch_size=1000, collate_fn=collate_tokens, num_workers=10
        )
        for i, batch in enumerate(dataloader):
            print(i)
