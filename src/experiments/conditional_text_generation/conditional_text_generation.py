import torch
from torch.optim import Adam

from src.pretrained_optimus.base_yelp import YelpPreTrainedOptimus
from src.utils.data.tokens import LabelledTokensBatch

from .cara import CARA


class YelpConditionalSentenceGenerator(YelpPreTrainedOptimus):
    def __init__(
        self,
        # YelpConditionalSentenceGenerator
        lr: float = 5e-5,
        eps: float = 1e-8,
        # PreTrainedOptimus, FineTunedOptimus, YelpPreTrainedOptimus
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.cara = CARA(
            self.encoder,
            self.decoder,
            self.tokeniser_encoder,
            self.tokeniser_decoder,
            self.decoder.config.n_embd,
            self.device,
            block_size=self.hparams.max_length,
        )

    def training_step(
        self,
        batch: LabelledTokensBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        attention_mask = (
            batch.tokens_batch.enc_tokens_batch
            == self.tokeniser_encoder.pad_token_id
        )
        loss_dict, acc_dict = self.cara(
            batch.tokens_batch.enc_tokens_batch,
            batch.tokens_batch.dec_tokens_batch,
            torch.tensor(batch.labels, device=self.device).long(),
            attention_mask,
        )
        loss_dict["loss"] = loss_dict["loss"].mean()
        loss_dict["loss_rec"] = loss_dict["loss_rec"].mean()
        self.log_dict({f"train/{k}": v.item() for k, v in loss_dict.items()})
        return loss_dict["loss"]

    def validation_step(
        self,
        batch: LabelledTokensBatch,
        batch_idx: int,
    ) -> None:
        attention_mask = (
            batch.tokens_batch.enc_tokens_batch
            == self.tokeniser_encoder.pad_token_id
        )
        result = self.cara(
            batch.tokens_batch.enc_tokens_batch,
            batch.tokens_batch.dec_tokens_batch,
            torch.tensor(batch.labels, device=self.device).long(),
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
        return Adam(
            self.cara.parameters(), lr=self.hparams.lr, eps=self.hparams.eps
        )
