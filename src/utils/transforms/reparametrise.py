import torch


def reparametrise(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = logvar.mul(0.5).exp()
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)
