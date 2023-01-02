import torch
import torch.nn as nn


class BinaryTextDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        mode: str = "mean",
        init_range: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, mode=mode, sparse=True
        )
        self.fc = nn.Linear(embed_dim, 1)
        # self._init_weights(init_range)

    # def _init_weights(self, init_range: float):
    #     self.embedding.weight.data.uniform_(-init_range, init_range)
    #     self.fc.weight.data.uniform_(-init_range, init_range)
    #     self.fc.bias.data.zero_()

    def forward(
        self, text_tokens: torch.FloatTensor, offsets: torch.LongTensor = None
    ) -> torch.FloatTensor:
        return self.fc(self.embedding(text_tokens, offsets))
