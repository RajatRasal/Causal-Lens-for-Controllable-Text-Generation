from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import PreTrainedTokenizer


# TODO: Change type to FloatTensor
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


def build_tokens(
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
