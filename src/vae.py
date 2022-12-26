import random
from collections import defaultdict
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from lightning_lite.utilities.seed import seed_everything
from torch import nn
from torch.utils.data import DataLoader
from transformers import (  # noqa: E501
    BertConfig,
    BertModel,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from .data import TokenisedSentencesFromIterable, collate_tokens
from .tokeniser import bert_pretrained_tokeniser, gpt2_pretrained_tokeniser
from .utils import beta_cycle_in_range


class BertGPT2VAE(pl.LightningModule):
    def __init__(
        self,
        tokeniser_encoder: BertTokenizer,
        tokeniser_decoder: GPT2Tokenizer,
        latent_size: int = 32,
        beta: float = 1.0,
        kl_threshold: float = 0.05,
        use_beta_schedule: bool = True,
        beta_cycle_len: int = 10,
        beta_cycle_ratio_increase: float = 0.25,
        beta_cycle_ratio_zero: float = 0.25,
        max_position_embeddings: int = 70,
    ):
        super().__init__()
        self.save_hyperparameters()

        # beta cycle
        self.beta_cycle = None

        # Encoder Model
        bert_config = BertConfig(
            max_position_embeddings=self.hparams.max_position_embeddings
        )
        self.model_encoder = BertModel(
            bert_config, self.hparams.tokeniser_encoder
        )

        # Decoder Model
        gpt2_config = GPT2Config(
            max_position_embeddings=self.hparams.max_position_embeddings
        )
        self.model_decoder = GPT2LMHeadModel(gpt2_config)
        self.model_decoder.resize_token_embeddings(
            len(self.hparams.tokeniser_decoder)
        )

        # Bottleneck
        self.latent_proj = nn.Linear(
            self.model_encoder.config.hidden_size,
            self.hparams.latent_size * 2,
            bias=False,
        )

        # Decoder memory embedding projection, different latent vector for
        # each layer
        self.memory_emb_flat = nn.Linear(
            self.hparams.latent_size,
            self.model_decoder.config.n_embd
            * self.model_decoder.config.n_layer,
            bias=False,
        )
        # self.emb_flat = nn.Linear(
        #     self.hparams.latent_size,
        #     self.model_decoder.config.n_embd,
        #     bias=False,
        # )

    def _reparametrise(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def encode(self, enc_tokens: torch.Tensor) -> torch.Tensor:
        # Masking
        attention_mask = (
            enc_tokens != self.hparams.tokeniser_encoder.pad_token_id
        ).float()
        # TODO: Figure our how items can be masked in the loss
        # dec_tokens[
        #     dec_tokens == self.hparams.tokeniser_decoder.pad_token_id
        # ] = -100

        # Encoding
        encoder_output = self.model_encoder(
            enc_tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        pooled_encoder_output = encoder_output.pooler_output

        # Bottleneck
        mean, logvar = self.latent_proj(pooled_encoder_output).chunk(2, -1)
        return self._reparametrise(mean, logvar), mean, logvar

    def _beta_from_schedule(self):
        if self.beta_cycle is None:
            assert self.trainer.max_steps > 0
            self.beta_cycle = beta_cycle_in_range(
                self.trainer.max_steps,
                start=0.0,
                stop=self.hparams.beta,
                n_cycle=self.hparams.beta_cycle_len,
                ratio_increase=self.hparams.beta_cycle_ratio_increase,
                ratio_zero=self.hparams.beta_cycle_ratio_zero,
            )
        return (
            1.0
            if self.global_step >= self.beta_cycle.shape[0]
            else self.beta_cycle[self.global_step]
        )

    def kl_loss(  # noqa: E501
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        loss_kl = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
        beta = (
            self._beta_from_schedule()
            if self.hparams.use_beta_schedule
            else self.hparams.beta
        )
        if beta == 0.0:
            kl_mask = (loss_kl > self.hparams.kl_threshold).float()
            loss_kl = (kl_mask * loss_kl).sum(dim=1)
        else:
            loss_kl = (loss_kl * beta).sum(dim=1)
        return loss_kl.mean()

    def _latent_to_past_key_values(
        self, latent: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        # [batch_size, n_embd * n_layer]
        memory_latent = self.memory_emb_flat(latent)
        memory_latent_per_layer = torch.split(
            memory_latent.unsqueeze(1), self.model_decoder.config.n_embd, dim=2
        )
        # [batch_size, num_heads, seq_length = 1, head_dim]
        # TODO: Remove magic numbers and replace with calculation
        past = [_l.view(-1, 12, 1, 64) for _l in memory_latent_per_layer]
        past_key_values = tuple(zip(past, past))
        return past_key_values

    def decode(
        self,
        dec_tokens: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        past_key_values = self._latent_to_past_key_values(latent)
        return self.model_decoder(
            dec_tokens,
            past_key_values=past_key_values,
            labels=dec_tokens,
            return_dict=True,
        )["loss"]

    def conditional_generation(
        self,
        latent: torch.Tensor,
        max_length: int,
        method: str = "top-p",
    ) -> List[str]:
        assert latent.shape[0] == 1, "Generate 1 sentence at a time."

        past_key_values = self._latent_to_past_key_values(latent)
        context_token = torch.tensor(
            self.hparams.tokeniser_decoder.encode(
                self.hparams.tokeniser_decoder.bos_token
            ),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        if method == "top-p":
            output_sequences = self.model_decoder.generate(
                input_ids=context_token,
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=True,
                num_return_sequences=3,
                top_k=0,
                top_p=0.9,
            )
            recons = self.hparams.tokeniser_decoder.batch_decode(
                output_sequences.tolist(),
                skip_special_tokens=True,
                clean_up_tokenisation_spaces=True,
            )
            return ["".join(recon) for recon in recons]
        else:
            raise Exception(f"Generation method {method} not known")

    def _step(
        self, enc_tokens: torch.Tensor, dec_tokens: torch.Tensor
    ) -> torch.Tensor:
        latent, mean, logvar = self.encode(enc_tokens)
        recon = self.decode(dec_tokens, latent)
        kl = self.kl_loss(mean, logvar)
        return recon + kl, recon, kl, latent

    def training_step(self, batch, batch_idx):
        elbo, recon, kl, _ = self._step(
            batch.enc_tokens_batch, batch.dec_tokens_batch
        )
        metrics = {"loss": elbo, "recon": recon, "kl": kl}
        self.log_dict({f"train/{k}": v.item() for k, v in metrics.items()})
        return metrics

    def validation_step(self, batch, batch_idx):
        elbo, recon_loss, kl_loss, latent = self._step(
            batch.enc_tokens_batch, batch.dec_tokens_batch
        )

        for i in range(len(batch.sentences)):
            # TODO: Include len calculation in dataset so we don't
            # need to recompute for each validation
            recons = self.conditional_generation(
                latent[i].unsqueeze(0),
                batch.dec_tokens_batch_lengths[i],
            )
            recons_str = "\n".join([recon_str for recon_str in recons])
            self.logger.experiment.add_text(
                batch.sentences[i], recons_str, self.global_step
            )

        return {"loss": elbo, "recon": recon_loss, "kl": kl_loss}

    def validation_epoch_end(self, outputs):
        collated_outputs = defaultdict(list)
        for step_output in outputs:
            for k, v in step_output.items():
                collated_outputs[k].append(v.item())

        self.log_dict(
            {f"val/{k}": sum(v) / len(v) for k, v in collated_outputs.items()}
        )

    def test_step(self, batch, batch_idx):
        elbo, recon_loss, kl_loss, latent = self._step(
            batch.enc_tokens_batch, batch.dec_tokens_batch
        )

        for i in range(len(batch.sentences)):
            # TODO: Include len calculation in dataset so we don't
            # need to recompute for each validation
            recons = self.conditional_generation(
                latent[i].unsqueeze(0),
                batch.dec_tokens_batch_lengths[i],
            )
            recons_str = "\n".join([recon_str for recon_str in recons])
            self.logger.experiment.add_text(
                batch.sentences[i], recons_str, self.global_step
            )

        return {"loss": elbo, "recon": recon_loss, "kl": kl_loss}

    def test_epoch_end(self, outputs):
        collated_outputs = defaultdict(list)
        for step_output in outputs:
            for k, v in step_output.items():
                collated_outputs[k].append(v.item())

        self.log_dict(
            {f"test/{k}": sum(v) / len(v) for k, v in collated_outputs.items()}
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5, eps=1e-8)


if __name__ == "__main__":
    # TODO: Output kl, loss, elbo etc. to Tensorboard
    # TODO: Interpolations
    # TODO: Currently we do not include an h_emb as seen here
    # https://github.com/ChunyuanLI/Optimus/blob/f63f4a7ca10aea022978500a37d72dd53a37a576/code/pytorch_transformers/modeling_gpt2.py#L472
    # We should still be able to get some good results since the paper
    # says the h_mem is more important.

    seed_everything(42, workers=True)

    # Training dataset
    tokeniser_encoder = bert_pretrained_tokeniser()
    tokeniser_decoder = gpt2_pretrained_tokeniser()
    train_list = load_dataset("wikitext", "wikitext-2-v1", cache_dir="./data")[
        "train"
    ]["text"]
    train_dataset = TokenisedSentencesFromIterable(
        train_list, tokeniser_encoder, tokeniser_decoder
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=5,
        collate_fn=collate_tokens,
        num_workers=32,
    )

    # Max epochs and steps needed for beta scheduler
    max_epochs = 1
    max_steps = max_epochs * len(train_dataset.it)

    # Validation dataset
    # TODO: We need 5x as much data.
    # Performance of LMs is linear in the log-scale of the no. of words.
    # https://aclanthology.org/2021.acl-long.90.pdf
    total_val_list = load_dataset(
        "wikitext", "wikitext-2-v1", cache_dir="./data"
    )["train"]["text"]
    val_sample = random.sample(total_val_list, 10)
    val_dataset = TokenisedSentencesFromIterable(
        val_sample, tokeniser_encoder, tokeniser_decoder
    )
    assert len(val_dataset.it) < 100
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=5,
        collate_fn=collate_tokens,
        num_workers=32,
    )

    # Defining the model
    model = BertGPT2VAE(tokeniser_encoder, tokeniser_decoder)

    # trainer = pl.Trainer(
    #     max_epochs=40,
    #     val_check_interval=100,
    #     accelerator="gpu", devices="-1", strategy="ddp"
    # )

    log_and_val_freq = 1000
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        val_check_interval=log_and_val_freq,
        log_every_n_steps=log_and_val_freq,
        accelerator="gpu",
        devices=[0],
        max_steps=max_steps,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
