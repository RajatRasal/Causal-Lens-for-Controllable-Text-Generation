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

        self.cara = CARA(
            self.encoder,
            self.decoder,
            self.tokeniser_encoder,
            self.tokeniser_decoder,
            768,
            self.device,
        )

    def training_step(
        self,
        batch: LabelledTokensBatch,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        attention_mask = (
            batch.tokens_batch.enc_tokens_batch
            == self.tokeniser_encoder.pad_token_id
        )
        loss_dict, acc_dict = self.cara(
            batch.tokens_batch.enc_tokens_batch,
            batch.tokens_batch.dec_tokens_batch,
            batch.labels,
            attention_mask,
        )
        self.log_dict({f"train/{k}": v.item() for k, v in loss_dict.items()})
        return loss_dict["loss"]

    def validation_step(
        self,
        batch: LabelledTokensBatch,
        batch_idx: int,
    ):
        attention_mask = (
            batch.tokens_batch.enc_tokens_batch
            == self.tokeniser_encoder.pad_token_id
        )
        result = self.cara(
            batch.tokens_batch.enc_tokens_batch,
            batch.tokens_batch.dec_tokens_batch,
            batch.labels,
            attention_mask,
        )
        if batch_idx == 0:
            generated = [self.untokenise(x) for x in result["generated"]]
            for orig, label, gen in zip(
                batch.tokens_batch.sentences, batch.labels, generated
            ):
                self.logger.experiment.add_text(
                    f"{label} - {orig}", gen, self.global_step
                )

    def configure_optimizers(self):
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
