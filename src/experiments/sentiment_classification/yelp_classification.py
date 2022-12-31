import multiprocessing
import random
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics.functional.classification import (  # noqa: E501
    binary_accuracy,
    binary_f1_score,
)

from src.pretrained_optimus.vae import PreTrainedOptimus
from src.utils.data.tokens import (  # noqa: E501
    LabelledTokensBatch,
    collate_labelled_tokens,
)

from .data import TokenisedSentencesYelpReviewPolarity


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


# TODO: Make this a generic parent class SentimentClassifier and make a Yelp
# child for this with the train_dataloader and val_dataloader implemented.
class YelpBinarySentimentClassifier(PreTrainedOptimus):
    def __init__(
        self,
        bert_model_name: str,
        gpt2_model_name: str,
        max_length: int = 64,
        use_freeze: bool = False,
        hidden_dropout_prob: float = 0.5,
        train_prop: float = 0.8,
        batch_size: int = 256,
        train_dataset_size: int = 10000,
        val_dataset_size: int = 1000,
        lr: int = 5e-5,
    ):
        self.save_hyperparameters()
        super().__init__(bert_model_name, gpt2_model_name)

        for param in self.decoder.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(self.hparams.hidden_dropout_prob)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def _log_metrics(
        self, metrics: Dict[str, torch.Tensor], prefix: str
    ) -> None:
        self.log_dict(
            {f"{prefix}/{k}": v.item() for k, v in metrics.items()},
            on_epoch=True,
            reduce_fx="mean",
        )

    def _step(
        self,
        enc_tokens: torch.FloatTensor,
        labels: torch.LongTensor,
        step_name: str,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(enc_tokens, labels=labels.float())
        preds = torch.round(F.sigmoid(outputs.logits)).long().flatten()

        # TODO: Confusion matrix
        metrics = {
            "loss": outputs.loss,
            "accuracy": binary_accuracy(preds, labels),
            "f1_score": binary_f1_score(preds, labels),
        }
        self._log_metrics(metrics, step_name)

        return metrics

    def training_step(
        self, batch: LabelledTokensBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self._step(
            batch.tokens_batch.enc_tokens_batch, batch.labels, "train"
        )

    def validation_step(
        self, batch: LabelledTokensBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self._step(
            batch.tokens_batch.enc_tokens_batch, batch.labels, "val"
        )

    def test_step(self, batch, batch_idx):
        return self._step(
            batch.tokens_batch.enc_tokens_batch, batch.labels, "test"
        )

    def configure_optimizers(self):
        encoder_parameters = [p for p in self.encoder.parameters()]
        classifier_parameters = [p for p in self.classifier.parameters()]
        return Adam(
            encoder_parameters + classifier_parameters,
            lr=self.hparams.lr,
        )

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ) -> ClassifierOutput:
        pooled_output = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )[1]

        if self.hparams.use_freeze:
            pooled_output = pooled_output.detach()

        logits = self.classifier(self.dropout(pooled_output))
        output = ClassifierOutput(logits=logits)

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1), labels.view(-1)
            )
            output.loss = loss

        return output
