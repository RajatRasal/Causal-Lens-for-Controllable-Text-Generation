import torch


def kl_divergence(
    mean: torch.FloatTensor,
    logvar: torch.FloatTensor,
    beta: float = 1.0,
    kl_threshold: float = 0.0,
) -> torch.FloatTensor:
    """
    inputs:
        mean: (B, N)
        logvar: (B, N)
        beta in [0, 1]
    output:
        kl divergence: (B,)
    """
    if not (0 <= beta <= 1.0):
        raise Exception("Beta must be between 0 and 1")
    loss_kl = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
    if kl_threshold > 0:
        loss_kl *= (loss_kl > kl_threshold).float()
    return beta * loss_kl.sum(dim=1)
