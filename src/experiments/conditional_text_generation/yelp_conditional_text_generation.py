import multiprocessing
import random
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.experiments.sentiment_classification.data import (
    TokenisedSentencesYelpReviewPolarity,
)
from src.pretrained_optimus.vae import PreTrainedOptimus
from src.utils.data.tokens import (  # noqa: E501
    LabelledTokensBatch,
    collate_labelled_tokens,
)

from .cara import CARA


@dataclass
class GeneratorOutput:
    """
    Args:
        gen (`torch.FloatTensor` of shape `(batch_size, IMG)`):
            The thing being generated.
        loss (`torch.FloatTensor` of shape `(1,)` when `labels` is provided):
            Classification loss.
    """

    gen: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class YelpConditionalSentenceGenerator(PreTrainedOptimus):
    def __init__(
        self,
        bert_model_name: str,
        gpt2_model_name: str,
        max_length: int = 64,
        train_prop: float = 0.8,
        batch_size: int = 256,
        train_dataset_size: int = 10000,
        val_dataset_size: int = 1000,
        discriminator_embed_dim: int = 32,
        decode_strategy: str = "greedy",
        lr: int = 5e-5,
    ):
        self.save_hyperparameters()
        super().__init__(bert_model_name, gpt2_model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False

        print(self.tokeniser_decoder.bos_token_id)
        self.cara = CARA(
            self.encoder,
            self.decoder,
            self.tokeniser_encoder,
            self.tokeniser_decoder,
            768,
            self.device,
        )

        # self.decoder_proj = nn.Linear(
        #     self.decoder.config.n_embd + 1,
        #     self.decoder.config.n_embd,
        # )
        # self.discriminator = BinaryTextDiscriminator(
        #     vocab_size=self.tokeniser_decoder.vocab_size,
        #     embed_dim=self.hparams.discriminator_embed_dim,
        # )

    # def forward(
    #     self,
    #     z: torch.FloatTensor,
    #     dec_tokens: torch.LongTensor,
    #     labels: torch.LongTensor,
    #     valid: torch.FloatTensor = None,
    #     freeze_generator: bool = False,
    # ) -> GeneratorOutput:
    #     print("Training mode:", self.decoder.training)
    #     z_cond = torch.cat((z, labels.unsqueeze(1).float()), dim=-1)
    #     z_cond = self.decoder_proj(z_cond)
    #     print('z_cond:', z_cond.requires_grad)
    #     # TODO: Implement batchwise conditional generation using
    #     # https://github.com/ChunyuanLI/Optimus/blob/f63f4a7ca10aea022978500a37d72dd53a37a576/code/examples/run_generation.py#L106
    #     # dec_tokens = self.decode(_z_cond, self.hparams.max_length, self.hparams.decode_strategy)

    #     res = self.decoder(
    #         input_ids=dec_tokens,
    #         past=z_cond,
    #         labels=dec_tokens,
    #         label_ignore=self.tokeniser_decoder.pad_token_id,
    #     )  # (B, seq_len, vocab_size)
    #     print('decoder_out:', res[0].requires_grad)  # , res[2].requires_grad)
    #     dec_tokens = res[1]
    #     print('decoder out:', dec_tokens.requires_grad)
    #     dec_tokens = torch.argmax(dec_tokens, dim=-1)  # (B, seq_len)
    #     print('decoder out:', dec_tokens.requires_grad)
    #     # (B, sentence_embed) - using nn.EmbeddingBag
    #     # (B, real or fake) = logits - using linear layer
    #     # logits = self.discriminator(_dec_tokens)

    #     # dec_tokens = [
    #     #     self.conditional_generation(_z_cond[i].unsqueeze(0))
    #     #     for i in range(_z_cond.size()[0])
    #     # ]

    #     # dec_tokens = []
    #     # lengths = [0]
    #     # for i in range(_z_cond.size()[0]):
    #     #     dec_tokens_i = self.conditional_generation(_z_cond[i].unsqueeze(0))
    #     #     dec_tokens.append(dec_tokens_i.squeeze(0))
    #     #     lengths.append(dec_tokens_i.size(0))
    #     # offsets = torch.tensor(lengths[:-1]).cumsum(dim=0).to(self.device)
    #     # text_list = torch.cat(dec_tokens)

    #     output = GeneratorOutput(gen=dec_tokens)

    #     if freeze_generator:
    #         dec_tokens = dec_tokens.detach()

    #     if valid is not None:
    #         logits = self.discriminator(dec_tokens)
    #         print('logits:', logits.requires_grad)
    #         # adversarial loss is binary cross-entropy
    #         loss = F.binary_cross_entropy_with_logits(
    #             logits.view(-1), valid.view(-1)
    #         )
    #         output.loss = loss

    #     return output

    # def _step(
    #     self,
    #     enc_tokens: torch.FloatTensor,
    #     dec_tokens: torch.FloatTensor,
    #     labels: torch.FloatTensor,
    #     optimizer_idx: int,
    #     step_name: str,
    # ) -> Dict[str, torch.Tensor]:
    #     # train generator
    #     if optimizer_idx == 0:
    #         # ground truth result
    #         valid = torch.ones(enc_tokens.size(0), 1).type_as(labels).to(self.device)

    #         # TODO: Might want to use reparametrised version?
    #         # generate images
    #         _, mean, _ = self.encode(enc_tokens)
    #         outputs = self.forward(mean.detach(), dec_tokens, labels, valid)

    #         # log sampled text
    #         for i, tokens in enumerate(outputs.gen[:6]):
    #             self.logger.experiment.add_text(
    #                 f"{step_name} / example {i}", self.untokenise(tokens), self.global_step
    #             )

    #         # loss
    #         self.log(f"{step_name}/g_loss", outputs.loss, prog_bar=True)
    #         return outputs.loss

    #     # train discriminator
    #     if optimizer_idx == 1:
    #         # Measure discriminator's ability to classify real from
    #         # generated samples

    #         # how well can it label as real?
    #         valid = torch.ones(dec_tokens.size(0), 1) \
    #             .type_as(labels) \
    #             .to(self.device)
    #         logits = self.discriminator(dec_tokens)
    #         real_loss = F.binary_cross_entropy_with_logits(
    #             logits.view(-1), valid.view(-1)
    #         )

    #         # how well can it label as fake?
    #         fake = torch.zeros(dec_tokens.size(0), 1) \
    #             .type_as(labels) \
    #             .to(self.device)
    #         z = torch.randn(
    #             dec_tokens.shape[0], self.hparams.latent_dim
    #         ).float().to(self.device)
    #         fake_outputs = self(z, labels, fake, freeze_generator=True)

    #         # discriminator loss is the average of these
    #         d_loss = (real_loss + fake_outputs.loss) / 2
    #         self.log(f"{step_name}/d_loss", d_loss, prog_bar=True)
    #         return d_loss

    # def _log_metrics(
    #     self, metrics: Dict[str, torch.Tensor], prefix: str
    # ) -> None:
    #     self.log_dict(
    #         {f"{prefix}/{k}": v.item() for k, v in metrics.items()},
    #         on_epoch=True,
    #         reduce_fx="mean",
    #     )

    def _step(
        self,
        enc_tokens_batch,
        dec_tokens_batch,
        cond_labels,
    ):
        attention_mask = enc_tokens_batch == self.tokeniser_encoder.pad_token_id
        loss_dict, _ = self.cara(
            enc_tokens_batch, dec_tokens_batch, cond_labels, attention_mask
        )
        return loss_dict

    def training_step(
        self,
        batch: LabelledTokensBatch,
        batch_idx: int,
        # optimizer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        return self._step(
            batch.tokens_batch.enc_tokens_batch,
            batch.tokens_batch.dec_tokens_batch,
            batch.labels,
            # optimizer_idx,
            # "train",
        )

    # def validation_step(
    #     self,
    #     batch: LabelledTokensBatch,
    #     batch_idx: int,
    # ) -> Dict[str, torch.Tensor]:
    #     return self._step(
    #         batch.tokens_batch.enc_tokens_batch,
    #         batch.tokens_batch.dec_tokens_batch,
    #         batch.labels,
    #         optimizer_idx,
    #         "val",
    #     )

    def configure_optimizers(self):
        # generator_parameters = [p for p in self.decoder.parameters()] + [p for p in self.decoder_proj.parameters()]
        # discriminator_parameters = [p for p in self.discriminator.parameters()]
        # opt_g = Adam(generator_parameters, lr=self.hparams.lr)
        # opt_d = Adam(discriminator_parameters, lr=self.hparams.lr)
        # return [opt_g, opt_d], []
        return Adam(self.cara.parameters(), lr=self.hparams.lr)

    def _dataset(self, split: str) -> TokenisedSentencesYelpReviewPolarity:
        return TokenisedSentencesYelpReviewPolarity(
            tokeniser_encoder=self.tokeniser_encoder,
            tokeniser_decoder=self.tokeniser_decoder,
            split=split,
            root="./data",
            max_length=self.hparams.max_length,
        )

    def _train_split_size(self, ds_len: int) -> int:
        return int(ds_len * self.hparams.train_prop)

    def train_dataloader(self) -> DataLoader:
        ds = self._dataset("train")
        train_max = self._train_split_size(len(ds))

        return DataLoader(
            ds,
            sampler=SubsetRandomSampler(
                random.sample(
                    range(train_max), self.hparams.train_dataset_size
                ),
            ),
            batch_size=self.hparams.batch_size,
            collate_fn=collate_labelled_tokens,
            num_workers=multiprocessing.cpu_count() - 1,
        )

    def val_dataloader(self) -> DataLoader:
        ds = self._dataset("train")
        ds_len = len(ds)
        train_max = self._train_split_size(ds_len)

        indices = range(train_max, ds_len)
        if self.hparams.val_dataset_size is None:
            val_dataset_size = len(indices)
        else:
            val_dataset_size = min(len(indices), self.hparams.val_dataset_size)

        return DataLoader(
            ds,
            sampler=SubsetRandomSampler(
                random.sample(indices, val_dataset_size),
            ),
            batch_size=self.hparams.batch_size,
            collate_fn=collate_labelled_tokens,
            num_workers=multiprocessing.cpu_count() - 1,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset("test"),
            batch_size=self.hparams.batch_size,
            collate_fn=collate_labelled_tokens,
            num_workers=multiprocessing.cpu_count() - 1,
        )
