import torch
from torch import nn


def batch_conditional_generation(
    decoder: nn.Module,
    input_ids: torch.LongTensor,
    z: torch.Tensor,
    max_length: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.LongTensor:
    assert input_ids.size()[0] == z.size()[0]
    generated = input_ids
    for _ in range(max_length):
        outputs = decoder(input_ids=generated, past=z)[0]
        next_token_logits = outputs[:, -1, :] / temperature
        next_token = torch.argmax(next_token_logits, dim=1)
        generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
    return generated
