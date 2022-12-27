import argparse
import random

import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.optimus.classify import BertClassifier
from src.optimus.data import (  # noqa: E501
    TokenisedSentencesYelpReviewPolarity,
    collate_yelp_tokens,
)
from src.optimus.vae import load_bert_gpt2_vae


def train(
    optimus_checkpoint_path: str,
    output_checkpoint_folder: str,
    dataset_size: int = 10000,
    batch_size: int = 128,
    max_epochs: int = 100,
    log_freq: int = 10,
    train_prop: float = 0.9,
):
    if optimus_checkpoint_path == output_checkpoint_folder:
        raise Exception("Output model path cannot be equal to input path")

    # TODO: Move dataloaders into the LightningModule so that we
    # can include batch_size as a hyperparameter
    classifier = BertClassifier(optimus_checkpoint_path)

    model = load_bert_gpt2_vae(optimus_checkpoint_path)

    ds = TokenisedSentencesYelpReviewPolarity(
        model.tokeniser_encoder,
        model.tokeniser_decoder,
        "train",
        "./data",
    )

    ds_len = len(ds)
    train_max = int(ds_len * train_prop)
    train_dataloader = DataLoader(
        ds,
        sampler=SubsetRandomSampler(
            random.sample(range(train_max), dataset_size)
        ),
        batch_size=batch_size,
        collate_fn=collate_yelp_tokens,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        ds,
        sampler=SubsetRandomSampler(
            random.sample(range(train_max, ds_len), 1000)
        ),
        batch_size=batch_size,
        collate_fn=collate_yelp_tokens,
        num_workers=8,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        val_check_interval=log_freq,
        log_every_n_steps=log_freq,
        accelerator="gpu",
        devices=[0],
        default_root_dir=output_checkpoint_folder,
    )

    trainer.fit(classifier, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-checkpoint-path", type=str, required=True
    )
    parser.add_argument(
        "-o",
        "--output-checkpoint-folder",
        type=str,
        default="./lightning_logs_classify_train",
    )
    parser.add_argument("-ds", "--dataset-size", type=int, default=10000)
    parser.add_argument("-e", "--max-epochs", type=int, default=100)
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-lf", "--log-freq", type=int, default=100)
    parser.add_argument("-p", "--train-prop", type=float, default=0.8)

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.test:
        pass
    else:
        train(
            args.input_checkpoint_path,
            args.output_checkpoint_folder,
            args.dataset_size,
            args.batch_size,
            args.max_epochs,
            args.log_freq,
            args.train_prop,
        )
