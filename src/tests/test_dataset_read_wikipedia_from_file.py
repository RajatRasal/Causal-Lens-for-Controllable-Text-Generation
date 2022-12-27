from torch.utils.data import DataLoader

from src.optimus.data import TokenisedSentencesFromFile, collate_tokens
from src.optimus.tokeniser import (  # noqa: E501
    bert_pretrained_tokeniser,
    gpt2_pretrained_tokeniser,
)

tokeniser_encoder = bert_pretrained_tokeniser()
tokeniser_decoder = gpt2_pretrained_tokeniser()

f = "./data/wikipedia.segmented.nltk.txt"
ds = TokenisedSentencesFromFile(f, tokeniser_encoder, tokeniser_decoder)

dataloader = DataLoader(
    ds, batch_size=10000, collate_fn=collate_tokens, num_workers=10
)

for i, x in enumerate(dataloader):
    print(x.enc_tokens_batch[0])
    if i % 100 == 0:
        print(i)
