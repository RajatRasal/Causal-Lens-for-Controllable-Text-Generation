from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ClassifierOutput:
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Classification scores before sigmoid.
        loss (`torch.FloatTensor` of shape `(1,)` when `labels` is provided):
            Classification loss.
    """

    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
