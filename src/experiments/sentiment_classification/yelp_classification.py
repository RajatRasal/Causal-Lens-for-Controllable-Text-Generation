from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.functional.classification import (  # noqa: E501
    binary_accuracy,
    binary_f1_score,
)

from src.pretrained_optimus.base_yelp import YelpPreTrainedOptimus
from src.utils.data.tokens import LabelledTokensBatch
from src.utils.outputs.classification import ClassifierOutput


class YelpBinarySentimentClassifier(YelpPreTrainedOptimus):
    def __init__(
        self,
        hidden_dropout_prob: float = 0.5,
        use_freeze: bool = False,
        # PreTrainedOptimus, FineTunedOptimus, YelpPreTrainedOptimus
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

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
        outputs = self(enc_tokens, labels=labels.float())
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.FloatTensor = None,
        att_mask: torch.BoolTensor = None,
    ) -> ClassifierOutput:
        pooled_output = self.encoder(input_ids, attention_mask=att_mask)[1]

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
