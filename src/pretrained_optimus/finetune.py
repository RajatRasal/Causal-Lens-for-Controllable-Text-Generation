from typing import Dict

import torch

from src.utils.data.tokens import TokensBatch
from src.utils.losses.kl import kl_divergence
from src.utils.scheduling.beta import beta_cycle_in_range

from .base import PreTrainedOptimus


class FineTunedOptimus(PreTrainedOptimus):
    def __init__(
        self,
        # FineTunedOptimus
        use_beta_schedule: bool = True,
        beta: float = 1.0,
        beta_cycle_len: int = 10,
        beta_cycle_ratio_increase: float = 0.25,
        beta_cycle_ratio_zero: float = 0.25,
        kl_threshold: float = 1.0,
        lr: float = 5e-5,
        eps: float = 1e-8,
        length_weighted: bool = True,
        # PreTrainedOptimus
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.beta_cycle = None

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, eps=self.hparams.eps
        )

    def _beta_from_schedule(self) -> float:
        if self.beta_cycle is None:
            max_steps = (
                len(self.train_dataloader) + 1
            ) * self.trainer.max_epochs
            self.beta_cycle = beta_cycle_in_range(
                max_steps,
                start=0.0,
                stop=self.hparams.beta,
                n_cycle=self.hparams.beta_cycle_len,
                ratio_increase=self.hparams.beta_cycle_ratio_increase,
                ratio_zero=self.hparams.beta_cycle_ratio_zero,
            )
        return self.beta_cycle[self.global_step]

    def _log_metrics(self, metrics: Dict[str, torch.FloatTensor], split: str):
        self.log_dict({f"{split}/{k}": v.item() for k, v in metrics.items()})

    def _step(
        self,
        enc_tokens: torch.FloatTensor,
        dec_tokens: torch.FloatTensor,
        enc_tokens_batch_lengths: torch.FloatTensor,
    ) -> Dict[str, torch.FloatTensor]:
        # Encoding text tokens
        z, mean, logvar = self.encode(enc_tokens)

        # Beta annealing
        beta = (
            self.hparams.beta
            if self.hparams.use_beta_schedule
            else self._beta_from_schedule()
        )

        # Loss calculation
        kl_loss = kl_divergence(mean, logvar, beta, self.hparams.kl_threshold)
        recon_loss = self.decode(z, dec_tokens)[0]

        # Scaling reconstruction loss by sentence length
        if self.hparams.length_weighted:
            recon_loss_weighted = recon_loss / enc_tokens_batch_lengths
        else:
            recon_loss_weighted = recon_loss
        loss = recon_loss_weighted + kl_loss

        return {
            "loss": loss.mean(),
            "recon_loss": recon_loss_weighted.mean(),
            "kl_loss": kl_loss.mean(),
        }

    def training_step(
        self, batch: TokensBatch, batch_idx: int
    ) -> torch.FloatTensor:
        metrics = self._step(
            batch.enc_tokens_batch,
            batch.dec_tokens_batch,
            torch.tensor(batch.enc_tokens_batch_lengths, device=self.device),
        )
        self._log_metrics(metrics, "train")
        return metrics

    def validation_step(self, batch: TokensBatch, batch_idx: int):
        metrics = self._step(
            batch.enc_tokens_batch,
            batch.dec_tokens_batch,
            torch.tensor(batch.enc_tokens_batch_lengths, device=self.device),
        )
        self._log_metrics(metrics, "val")
        return metrics
