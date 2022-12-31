from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.optimus.tokeniser import (  # noqa: E501
    bert_pretrained_tokeniser,
    gpt2_pretrained_tokeniser,
)

from .arch import BertForLatentConnector, GPT2ForLatentConnector
from .decoding import top_k_top_p_filtering


# TODO: Copy more methods from arch.utils to here with tests
# TODO: For each experiment, we can subclass this class.
class PreTrainedOptimus(pl.LightningModule):
    """
    A VAE for controllable language generation where the encoder uses the BERT
    architecture and the decoder uses the GPT2 architecture, both with the
    default hyperparameter settings.
    """

    def __init__(
        self,
        bert_model_name: str,
        gpt2_model_name: str,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokeniser_encoder = bert_pretrained_tokeniser()
        self.tokeniser_decoder = gpt2_pretrained_tokeniser()

        self.encoder = BertForLatentConnector.from_pretrained(
            self.hparams.bert_model_name
        )
        self.decoder = GPT2ForLatentConnector.from_pretrained(
            self.hparams.gpt2_model_name
        )

        self.CONTEXT_TOKEN = torch.tensor(
            [[self.tokeniser_decoder.bos_token_id]],
            dtype=torch.long,
            device=self.device,
        )

    def tokenise(
        self, x: List[str], return_type: Optional[str] = "pt"
    ) -> torch.Tensor:
        return self.tokeniser_encoder(x, return_tensors=return_type)[
            "input_ids"
        ]

    def untokenise(self, tokens: torch.Tensor) -> List[str]:
        out = self.tokeniser_decoder.decode(
            tokens.tolist(),
            clean_up_tokenization_spaces=True,
        )
        return " ".join(out.split()[1:-1])

    def reparametrise(
        sef, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(tokens, attention_mask=(tokens > 0).float())[1]
        mean, logvar = self.encoder.linear(encoding).chunk(2, -1)
        return self.reparametrise(mean, logvar), mean, logvar

    def decode(self, z: torch.Tensor, labels: torch.Tensor):
        return self.decoder(
            input_ids=labels,
            past=z,
            labels=labels,
            label_ignore=self.pad_token_id,
        )

    def conditional_generation(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        generated = self.CONTEXT_TOKEN
        next_token_id = None
        while next_token_id != self.tokeniser_decoder.eos_token_id:
            outputs = self.decoder(input_ids=generated, past=z)[0]
            next_token_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1), num_samples=1
            )
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            next_token_id = next_token.unsqueeze(0)[0, 0].item()
        return generated

    def reconstruct(
        self, source_sent: str, deterministic: bool = True, **dec_kwargs
    ) -> str:
        if deterministic:
            _, z, _ = self.encode(self.tokenise([source_sent])[0].unsqueeze(0))
            tokens = self.conditional_generation(z, **dec_kwargs).squeeze(0)
            return self.untokenise(tokens)
        else:
            # TODO: Use reparametrised vector as opposed to z
            raise NotImplementedError("Deterministic must be true")