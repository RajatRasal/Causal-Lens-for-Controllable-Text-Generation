import multiprocessing
import random

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.utils.data.tokens import LabelledTokensBatch, collate_labelled_tokens
from src.utils.data.yelp_dataset import TokenisedSentencesYelpReviewPolarity

from .finetune import FineTunedOptimus


class YelpPreTrainedOptimus(FineTunedOptimus):
    def __init__(
        self,
        # PreTrainedOptimus
        pretrained_latent_dim: int,
        pretrained_beta: float,
        pretrained_dataset: str,
        # FineTunedOptimus
        use_beta_schedule: bool = True,
        beta: float = 1.0,
        beta_cycle_len: int = 10,
        beta_cycle_ratio_increase: float = 0.25,
        beta_cycle_ratio_zero: float = 0.25,
        # YelpPreTrainedOptimus
        max_length: int = 64,
        batch_size: int = 256,
        train_prop: float = 0.8,
        train_dataset_size: int = 10000,
        val_dataset_size: int = 1000,
        storage_root: str = "./data",
    ):
        super().__init__(
            pretrained_latent_dim, pretrained_beta, pretrained_dataset
        )
        self.save_hyperparameters()

        self._ds = {}

    def _dataset(self, split: str) -> TokenisedSentencesYelpReviewPolarity:
        if split in self._ds:
            return self._ds[split]

        ds = TokenisedSentencesYelpReviewPolarity(
            tokeniser_encoder=self.tokeniser_encoder,
            tokeniser_decoder=self.tokeniser_decoder,
            split=split,
            root=self.hparams.storage_root,
            max_length=self.hparams.max_length,
        )
        self._ds[split] = ds
        return ds

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

    def training_step(
        self, batch: LabelledTokensBatch, batch_idx: int
    ) -> torch.FloatTensor:
        return super().training_step(batch.tokens_batch, batch_idx)

    def validation_step(
        self, batch: LabelledTokensBatch, batch_idx: int
    ) -> None:
        super().validation_step(batch.tokens_batch, batch_idx)
