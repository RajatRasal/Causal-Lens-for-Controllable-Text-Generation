import multiprocessing
import random
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_lite.utilities.seed import seed_everything
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.optimus.data import (  # noqa: E501
    LabelledTokensBatch,
    TokenisedSentencesYelpReviewPolarity,
    collate_labelled_tokens,
)

from .vae import PreTrainedOptimus


# TODO: Make this a generic parent class SentimentClassifier and make a Yelp
# child for this with the train_dataloader and val_dataloader implemented.
class YelpSentimentClassifier(PreTrainedOptimus):
    def __init__(
        self,
        bert_model_name: str,
        gpt2_model_name: str,
        max_length: int = 64,
        use_freeze: bool = False,
        hidden_dropout_prob: float = 0.5,
        num_labels: int = 2,
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
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            self.hparams.num_labels,
        )

    def _log_metrics(
        self, metrics: Dict[str, torch.Tensor], prefix: str
    ) -> None:
        self.log_dict(
            {f"{prefix}/{k}": v.item() for k, v in metrics.items()},
        )

    def _step(
        self, enc_tokens: torch.Tensor, labels: torch.Tensor, step_name: str
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(enc_tokens, labels=labels)
        # TODO: Calculate accuracy
        # TODO: Confusion matrix
        # TODO: ROC AUC
        metrics = {"loss": outputs[0][0]}
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

        return DataLoader(
            ds,
            sampler=SubsetRandomSampler(
                random.sample(
                    range(train_max, ds_len), self.hparams.val_dataset_size
                ),
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
    ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        if self.hparams.use_freeze:
            pooled_output = pooled_output.detach()

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.hparams.num_labels == 1:
                #  We are doing regression
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.hparams.num_labels),
                    labels.view(-1),
                )
            outputs = (loss,) + outputs

        # TODO: Return a dataclass, possibly from the Huggingface library.
        # (loss), logits, (hidden_states), (attentions)
        return outputs, pooled_output


if __name__ == "__main__":
    seed_everything(42)

    classifier = YelpSentimentClassifier(
        "bert-optimus-cased-snli-latent-768-beta-1",
        "gpt2-optimus-cased-snli-beta-1",
        val_dataset_size=1000,
        train_dataset_size=10000,
        batch_size=128,
    )

    log_freq = 50
    trainer = pl.Trainer(
        max_epochs=100,
        val_check_interval=log_freq,
        log_every_n_steps=log_freq,
        accelerator="gpu",
        devices=[0],
        default_root_dir="./lightning_logs_classify_train",
    )
    trainer.fit(classifier)
