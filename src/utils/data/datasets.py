from typing import Iterator, List

from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from .tokens import Tokens, build_tokens


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
        return build_tokens(
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
