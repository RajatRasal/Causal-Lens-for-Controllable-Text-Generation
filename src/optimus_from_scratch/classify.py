import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .vae import load_bert_gpt2_vae


class BertClassifier(pl.LightningModule):
    def __init__(self, checkpoint_path: str, classes: int = 2):
        super().__init__()
        self.save_hyperparameters()

        model = load_bert_gpt2_vae(checkpoint_path)
        self.bert = model.model_encoder
        self.pad_token_id = model.tokeniser_encoder.pad_token_id
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            self.hparams.classes,
        )

    def _features(self, tokens):
        return self.bert(
            tokens,
            attention_mask=(tokens != self.pad_token_id).float(),
            output_hidden_states=True,
        ).pooler_output

    def _step(self, batch, log_str):
        latent = self._features(batch.tokens_batch.enc_tokens_batch)
        outputs = self.classifier(latent)
        if self.hparams.classes == 2:
            logits = F.log_softmax(outputs, dim=1)
        else:
            logits = F.log_sigmoid(outputs)
        loss = F.nll_loss(logits, batch.labels)

        preds = torch.argmax(torch.exp(logits), dim=1)
        acc = (batch.labels == preds.flatten()).sum() / len(batch.labels)

        metrics = {"loss": loss, "accuracy": acc}
        self.log_dict({f"{log_str}/{k}": v.item() for k, v in metrics.items()})

        return loss

    def training_step(self, batch, idx):
        return self._step(batch, "train")

    def validation_step(self, batch, idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
