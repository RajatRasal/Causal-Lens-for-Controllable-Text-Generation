import torch
import torch.nn.functional as F


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: int = 0.0,
    filter_value: float = -float("Inf"),
):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p)
    filtering

    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability
            (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p
            (nucleus filtering).
            Nucleus filtering is described in Holtzman et al.
            (http://arxiv.org/abs/1904.09751)

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1

    # Safety check
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the
        # top-k
        _threshold = torch.topk(logits, top_k)[0][..., -1, None]
        indices_to_remove = logits < _threshold
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        softmax_sorted_logits = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(softmax_sorted_logits, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        remove_mask = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above
        # the threshold
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = 0

        indices_to_remove = sorted_indices[remove_mask]
        logits[indices_to_remove] = filter_value

    return logits
