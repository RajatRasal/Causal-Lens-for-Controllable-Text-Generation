import torch
import torch.nn as nn


class MeanLayer(nn.Module):
    def __init__(self, dim: int = -1):
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=self.dim)


class TransposeLayer(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)
