import re

from torch.utils.data import Dataset
from torchtext.datasets import YelpReviewPolarity
from transformers import PreTrainedTokenizer

from src.utils.data.tokens import LabelledTokens, build_tokens


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
                tokens=build_tokens(
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
            return self._cache[idx]
