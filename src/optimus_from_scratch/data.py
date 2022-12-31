import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Union

import torch
from torch.utils.data import Dataset, IterableDataset
from torchtext.datasets import YelpReviewPolarity
from transformers import PreTrainedTokenizer


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


@dataclass
class LabelledTokens:
    tokens: Tokens
    label: int
    original_doc_id: int


@dataclass
class LabelledTokensBatch:
    tokens_batch: TokensBatch
    labels: List[int]


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


def collate_labelled_tokens(
    tokens_list: List[LabelledTokens],
) -> LabelledTokensBatch:
    tokens = []
    labels = []

    for token in tokens_list:
        tokens.append(token.tokens)
        labels.append(token.label)

    return LabelledTokensBatch(
        tokens_batch=collate_tokens(tokens),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def _build_tokens(
    line: str,
    tokeniser_encoder: PreTrainedTokenizer,
    tokeniser_decoder: PreTrainedTokenizer,
    tokeniser_kwargs: Dict[str, Union[str, bool, int]],
) -> Tokens:
    # TODO: Create our own Tokeniser wrapper which can do the
    # necesary preprocessing!!
    # Currently this assume that tokeniser_decoder is the gpt2 decoder!!!!
    dec_sentence = (
        tokeniser_decoder.bos_token + line + tokeniser_decoder.eos_token
    )
    enc_tokens = tokeniser_encoder.encode(
        text=line,
        **tokeniser_kwargs,
    )
    enc_tokens_length = (
        (enc_tokens != tokeniser_encoder.pad_token_id).sum().item()
    )
    dec_tokens = tokeniser_decoder.encode(
        text=dec_sentence,
        **tokeniser_kwargs,
    )
    dec_tokens_length = (
        (dec_tokens != tokeniser_decoder.pad_token_id).sum().item()
    )
    return Tokens(
        enc_tokens=enc_tokens,
        enc_tokens_length=enc_tokens_length,
        dec_tokens=dec_tokens,
        dec_tokens_length=dec_tokens_length,
        sentence=line,
    )


class TokenisedSentences(IterableDataset):

    TOKENISER_KWARGS = {
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
        "max_length": 64,
    }

    def __init__(
        self,
        tokeniser_encoder: PreTrainedTokenizer,
        tokeniser_decoder: PreTrainedTokenizer,
    ):
        self.tokeniser_encoder = tokeniser_encoder
        self.tokeniser_decoder = tokeniser_decoder

    def build_tokens(self, line: str) -> Tokens:
        return _build_tokens(
            line,
            self.tokeniser_encoder,
            self.tokeniser_decoder,
            self.TOKENISER_KWARGS,
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
        self.it = self._filter_sentences(iterable)

    def _filter_sentences(self, lines: List[str]) -> List[str]:
        all_sents = []
        for line in lines:
            # Some lines are blank or contain titles such as
            #  = = Cricket = =
            # which can be picked up with the startswith.
            if not line or line.startswith(" = "):
                continue
            # Split sentences
            # line[:-1] == "\n"
            # line[:-2] == " "
            all_sents.extend(
                [sent + "." for sent in line[:-2].split(".") if sent]
            )
        return all_sents

    def __iter__(self) -> Iterator[Tokens]:
        for sent in self.it:
            tokens = self.build_tokens(sent)
            if tokens.enc_tokens_length <= 64:
                yield tokens


class TokenisedSentencesYelpReviewPolarity(Dataset):

    LOOKUP_POLARITY = {1: 0, 2: 1}

    def __init__(
        self,
        tokeniser_encoder: PreTrainedTokenizer,
        tokeniser_decoder: PreTrainedTokenizer,
        split: str,
        root: str,
        max_length: int = 100,
        return_tensors: str = "pt",
        truncation: bool = True,
        padding: str = "max_length",
    ):
        self.split = split
        self.root = root
        self.yelp_review_dataset = YelpReviewPolarity(
            root=self.root, split=self.split
        )
        self.tokeniser_encoder = tokeniser_encoder
        self.tokeniser_decoder = tokeniser_decoder

        self.max_length = max_length
        self.return_tensors = return_tensors
        self.truncation = truncation
        self.padding = padding

        self._reviews_to_sentences()
        self.n_sents = len(self.sentences)

        self._cache = {}

    def _reviews_to_sentences(self):
        self.sentences = []
        self.polarity = []
        self.original_review = []
        for i, (polarity, review) in enumerate(self.yelp_review_dataset):
            sentences = re.split("\\. *", review.replace("\\n", ""))
            for sentence in sentences:
                if not sentence or not re.search("[a-zA-Z0-9]", sentence):
                    continue
                sentence += "."
                if len(sentence.split(" ")) >= self.max_length:
                    continue
                self.sentences.append(sentence)
                self.polarity.append(self.LOOKUP_POLARITY[polarity])
                self.original_review.append(i)

    def __len__(self) -> int:
        return self.n_sents

    def __getitem__(self, idx) -> LabelledTokens:
        if idx not in self._cache:
            token = LabelledTokens(
                tokens=_build_tokens(
                    self.sentences[idx],
                    self.tokeniser_encoder,
                    self.tokeniser_decoder,
                    {
                        "max_length": self.max_length,
                        "return_tensors": self.return_tensors,
                        "truncation": self.truncation,
                        "padding": self.padding,
                    },
                ),
                label=self.polarity[idx],
                original_doc_id=self.original_review[idx],
            )
            self._cache[idx] = token
            return token
        else:
            print(f"Found {idx}")
            return self._cache[idx]
