from typing import List, Tuple

import pytorch_lightning as pl
import torch
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

from .data import TokenisedSentences, collate_tokens
from .tokeniser import bert_pretrained_tokeniser, gpt2_pretrained_tokeniser


class BertGPT2VAE(pl.LightningModule):
    def __init__(
        self,
        tokeniser_encoder: BertTokenizer,
        tokeniser_decoder: GPT2Tokenizer,
        latent_size: int = 32,
        beta: float = 1.0,
        kl_threshold: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder Model
        bert_config = BertConfig()
        self.model_encoder = BertModel(
            bert_config, self.hparams.tokeniser_encoder
        )

        # Decoder Model
        gpt2_config = GPT2Config()
        self.model_decoder = GPT2LMHeadModel(
            gpt2_config
        )  # , self.hparams.tokeniser_decoder)
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

    def kl_loss(  # noqa: E501
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        loss_kl = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
        kl_mask = (loss_kl > self.hparams.kl_threshold).float()
        return (kl_mask * loss_kl).sum() * self.hparams.beta

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
        assert latent.shape[0] == 1

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
        return {"loss": elbo, "recon": recon, "kl": kl}

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
            print()
            print(f"ORIGINAL:\n{batch.sentences[i]}")
            print()
            print(f"RECONSTRUCTION:\n{recons_str}")
            print()

        return {"loss": elbo, "recon": recon_loss, "kl": kl_loss}

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
            print()
            print(f"ORIGINAL:\n{batch.sentences[i]}")
            print()
            print(f"RECONSTRUCTION:\n{recons_str}")
            print()

        return {"loss": elbo, "recon": recon_loss, "kl": kl_loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5, eps=1e-8)


if __name__ == "__main__":
    # TODO: Output kl, loss, elbo etc. to Tensorboard
    # TODO: Output 1000 example sentences to Tensorboard
    # TODO: Interpolations
    # TODO: Varying beta during training runs
    # TODO: Use Wikitext-2 from huggingface datasets - smaller dataset,
    # eaiser to play with.
    # TODO: Currently we do not include an h_emb as seen here
    # https://github.com/ChunyuanLI/Optimus/blob/f63f4a7ca10aea022978500a37d72dd53a37a576/code/pytorch_transformers/modeling_gpt2.py#L472
    # We should still be able to get some good results since the paper
    # says the h_mem is more important.

    seed_everything(42)

    # Dataset and dataloader
    tokeniser_encoder = bert_pretrained_tokeniser()
    tokeniser_decoder = gpt2_pretrained_tokeniser()
    file = "./data/wikipedia.segmented.nltk.txt"
    dataset = TokenisedSentences(file, tokeniser_encoder, tokeniser_decoder)
    train_dataloader = DataLoader(
        dataset,
        batch_size=5,
        collate_fn=collate_tokens,
        num_workers=32,
    )

    # Defining the model
    model = BertGPT2VAE(tokeniser_encoder, tokeniser_decoder)

    # Some lines for verification
    lines = [
        dataset.build_tokens("the little girl plays with the toys."),
        dataset.build_tokens("A girl makes a silly face."),
        dataset.build_tokens("People are walking near a road."),
    ]
    val_dataloader = DataLoader(  # noqa: E501
        lines,
        batch_size=5,
        collate_fn=collate_tokens,
    )

    # trainer = pl.Trainer(
    #     max_epochs=40,
    #     val_check_interval=100,
    #     accelerator="gpu", devices="-1", strategy="ddp"
    # )
    trainer = pl.Trainer(
        max_epochs=1,
        val_check_interval=10,
        log_every_n_steps=100,
        accelerator="gpu",
        devices=[0],
    )

    trainer.fit(model, train_dataloader, val_dataloader)
