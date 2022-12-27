import re

from tqdm import tqdm

from src.optimus.data import TokenisedSentencesYelpReviewPolarity
from src.optimus.tokeniser import (  # noqa: E501
    bert_pretrained_tokeniser,
    gpt2_pretrained_tokeniser,
)

tokeniser_encoder = bert_pretrained_tokeniser()
tokeniser_decoder = gpt2_pretrained_tokeniser()

ds = TokenisedSentencesYelpReviewPolarity(
    tokeniser_encoder, tokeniser_decoder, "test", "./data", max_length=64
)

assert ds.n_sents == len(ds.original_review)
assert ds.n_sents == len(ds.polarity)

for row in tqdm(ds):
    assert row.tokens.sentence
    assert row.tokens.sentence[0] != " "
    assert "\n" not in row.tokens.sentence
    assert re.search("[a-zA-Z0-9]", row.tokens.sentence), row.tokens.sentence
