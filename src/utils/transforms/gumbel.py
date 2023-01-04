import torch
import torch.nn.functional as F


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [..., n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [..., n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it
        will be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)  # (..., n_class)

    if hard:  # return onehot
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)  # one hot
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y = (y_hard - y).detach() + y

    return y  # (..., n_class)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return F.softmax(y / temperature, dim=-1)


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape).to(device=device)
    return -torch.log(-torch.log(U + eps) + eps)
