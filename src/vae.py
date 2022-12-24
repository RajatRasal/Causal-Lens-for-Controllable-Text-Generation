import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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

    def _reparametrise(self, mean: torch.Tensor, logvar: torch.Tensor):
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def _step(self, enc_tokens: torch.Tensor, dec_tokens: torch.Tensor):
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
        latent = self._reparametrise(mean, logvar)

        # [batch_size, n_embd * n_layer]
        memory_latent = self.memory_emb_flat(latent)
        memory_latent_per_layer = torch.split(
            memory_latent.unsqueeze(1), self.model_decoder.config.n_embd, dim=2
        )

        # [batch_size, num_heads, seq_length = 1, head_dim]
        # TODO: Remove magic numbers and replace with calculation
        m = [_l.view(-1, 12, 1, 64) for _l in memory_latent_per_layer]
        m = tuple(zip(m, m))

        # Decoding
        outputs = self.model_decoder(
            dec_tokens,
            past_key_values=m,
            labels=dec_tokens,
            return_dict=True,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch.enc_tokens_batch, batch.dec_tokens_batch)[
            "loss"
        ]

    def validation_step(self, batch, batch_idx):
        logits = self._step(
            batch.enc_tokens_batch, batch.dec_tokens_batch
        ).logits
        softmax = F.softmax(logits)
        argmax = torch.argmax(softmax, dim=2)
        recons = self.hparams.tokeniser_decoder.batch_decode(
            argmax.tolist(),
            skip_special_tokens=True,
            clean_up_tokenisation_spaces=True,
        )
        for orig, recon in zip(batch.sentences, recons):
            print()
            print("Original:", orig)
            print("Reconstr:", "".join(recon))
            print()

    def test_step(self, batch, batch_idx):
        enc_tokens = batch.enc_tokens_batch
        dec_tokens = batch.dec_tokens_batch

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
        latent = self._reparametrise(mean, logvar) * 5

        # [batch_size, n_embd * n_layer]
        memory_latent = self.memory_emb_flat(latent)
        memory_latent_per_layer = torch.split(
            memory_latent.unsqueeze(1), self.model_decoder.config.n_embd, dim=2
        )

        # [batch_size, num_heads, seq_length = 1, head_dim]
        # TODO: Remove magic numbers and replace with calculation
        m = [_l.view(-1, 12, 1, 64) for _l in memory_latent_per_layer]
        m = tuple(zip(m, m))

        output_sequences = self.model_decoder.generate(
            input_ids=dec_tokens,
            past_key_values=m,
            max_length=(
                dec_tokens != self.hparams.tokeniser_decoder.pad_token_id
            ).sum(),
            do_sample=True,
            num_return_sequences=3,
            top_p=0.9,
        )
        recons = self.hparams.tokeniser_decoder.batch_decode(
            output_sequences.tolist(),
            skip_special_tokens=True,
            clean_up_tokenisation_spaces=True,
        )
        for orig, recon in zip(batch.sentences, recons):
            print()
            print("Original:", orig)
            print("Reconstr:", "".join(recon))
            print()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)


if __name__ == "__main__":
    # TODO: Output kl, loss, elbo etc. to Tensorboard
    # TODO: Output 1000 example sentences to Tensorboard
    # TODO: Example interpolation from paper - get z for default sentence then
    # vary it.
    # TODO: Varying beta during training runs
    # TODO: Create a dataset for BERTTokenised sentences, create another for
    # GPTTokenised sentences
    # Each dataset will cache the tokens in a folder, and include a link to
    # the original source file in that folder. The tokens in the folder will
    # be in chunked files.
    # TODO: Include attention mask?
    # TODO: Use Wikitext-2 from huggingface datasets - smaller dataset,
    # eaiser to play with.
    # TODO: Move build_tokens to

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
        dataset.build_tokens("the children are watching a show."),
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
        max_epochs=1, val_check_interval=100, accelerator="gpu", devices=[1]
    )

    trainer.fit(model, train_dataloader, val_dataloader)
