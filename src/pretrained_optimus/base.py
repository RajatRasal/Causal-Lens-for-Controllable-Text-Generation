from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.utils.decoder.top_k_top_p_filtering import top_k_top_p_filtering
from src.utils.tokeniser.tokeniser import (  # noqa: E501
    bert_pretrained_tokeniser,
    gpt2_pretrained_tokeniser,
)
from src.utils.transforms.reparametrise import reparametrise

from .arch import BertForLatentConnector, GPT2ForLatentConnector


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
        pretrained_latent_dim: int = 32,
        pretrained_beta: float = 1.0,
        pretrained_dataset: str = None,
    ):
        if pretrained_latent_dim not in [32, 768]:
            raise ValueError(
                f"pretrained_latent_dim must be either 32 or 768, not {pretrained_latent_dim}"  # noqa: E501
            )

        if pretrained_beta not in [0.0, 0.5, 1.0]:
            raise ValueError(
                f"pretrained_beta must be 0.0, 0.5 or 1.0, not {pretrained_beta}"  # noqa: E501
            )

        if pretrained_dataset not in ["snli", "wiki", None]:
            raise ValueError(
                f"pretrained_dataset must be snli, wiki or NONE, not {pretrained_dataset}"  # noqa: E501
            )

        super().__init__()
        self.save_hyperparameters()

        self.tokeniser_encoder = bert_pretrained_tokeniser()
        self.tokeniser_decoder = gpt2_pretrained_tokeniser()

        model_name = f"latent-{self.hparams.pretrained_latent_dim}"
        model_name += f"-beta-{self.hparams.pretrained_beta}"
        if self.hparams.pretrained_dataset is not None:
            model_name += f"-dataset-{self.hparams.pretrained_dataset}"

        self.encoder = BertForLatentConnector.from_pretrained(
            f"bert-optimus-cased-{model_name}"
        )
        self.decoder = GPT2ForLatentConnector.from_pretrained(
            f"gpt2-optimus-cased-{model_name}"
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

    def encode(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        encoding = self.encoder(tokens, attention_mask=(tokens > 0).float())[1]
        mean, logvar = self.encoder.linear(encoding).chunk(2, -1)
        return reparametrise(mean, logvar), mean, logvar

    def decode(self, z: torch.Tensor, labels: torch.Tensor):
        return self.decoder(
            input_ids=labels,
            past=z,
            labels=labels,
            label_ignore=self.tokeniser_decoder.pad_token_id,
        )

    def conditional_generation(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        max_length: int = 64,
    ) -> torch.Tensor:
        # TODO: Implement batch decodings from CARA.
        # TODO: Implement this in a separate method under utils/.
        """
        Strategy = top_k or top_p for 1 latent vector
        """
        assert len(z.size()) == 2 and z.size()[0] == 1
        generated = self.CONTEXT_TOKEN.to(self.device)
        next_token_id = None
        count = 0
        while (
            next_token_id != self.tokeniser_decoder.eos_token_id
            and count != max_length
        ):
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
            count += 1
        return generated

    def sample_decode(
        self,
        z: torch.Tensor,
        max_length: int,
        strategy: str = "greedy",
    ) -> torch.LongTensor:
        # TODO: Implement this in a separate method under utils/.
        """
        z: size B x D
        """
        generated = self.CONTEXT_TOKEN.repeat(z.size()[0], 1).to(self.device)
        for _ in range(max_length):
            outputs = self.decoder(input_ids=generated, past=z)[0]
            next_token_logits = outputs[:, -1, :]
            if strategy == "greedy":
                next_token = torch.argmax(next_token_logits, dim=1)
            else:
                probs = F.softmax(next_token_logits, dim=1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
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
