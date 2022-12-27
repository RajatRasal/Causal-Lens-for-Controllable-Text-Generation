import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .data import TokenisedSentences, collate_tokens
from .tokeniser import bert_pretrained_tokeniser, gpt2_pretrained_tokeniser
from .vae import BertGPT2VAE

tokeniser_encoder = bert_pretrained_tokeniser()
tokeniser_decoder = gpt2_pretrained_tokeniser()

file = "./data/wikipedia.segmented.nltk.txt"
dataset = TokenisedSentences(file, tokeniser_encoder, tokeniser_decoder)
lines = [
    dataset.build_tokens("The little girl plays with the toys."),
    dataset.build_tokens("A girl makes a silly face."),
    dataset.build_tokens("People are walking near a road."),
]
dataloader = DataLoader(  # noqa: E501
    lines,
    batch_size=1,
    collate_fn=collate_tokens,
)

model = BertGPT2VAE.load_from_checkpoint(
    checkpoint_path="./lightning_logs/version_19/checkpoints/epoch=0-step=59000.ckpt",  # noqa: E501
    map_location=None,
)
model.tokeniser_decoder = tokeniser_decoder
model.tokeniser_encoder = tokeniser_encoder

trainer = pl.Trainer(accelerator="gpu", devices=[2])
trainer.test(model, dataloaders=dataloader)
