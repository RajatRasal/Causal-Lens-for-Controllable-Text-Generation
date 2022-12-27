from datasets import load_dataset
from torch.utils.data import DataLoader

from src.optimus.data import TokenisedSentencesFromIterable, collate_tokens
from src.optimus.tokeniser import (  # noqa: E501
    bert_pretrained_tokeniser,
    gpt2_pretrained_tokeniser,
)

tokeniser_encoder = bert_pretrained_tokeniser()
tokeniser_decoder = gpt2_pretrained_tokeniser()

iterable = load_dataset("wikitext", "wikitext-2-v1", cache_dir="./data")[
    "train"
]["text"]
dataset = TokenisedSentencesFromIterable(
    iterable, tokeniser_encoder, tokeniser_decoder
)
dataloader = DataLoader(
    dataset, batch_size=1000, collate_fn=collate_tokens, num_workers=10
)
for i, batch in enumerate(dataloader):
    print(i)
