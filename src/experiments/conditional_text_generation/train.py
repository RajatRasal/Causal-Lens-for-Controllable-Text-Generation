import argparse

import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

from .yelp_conditional_text_generation import YelpConditionalSentenceGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-dataset-size", type=int, default=10000)
    parser.add_argument("--val-dataset-size", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=100)
    # 128 works fine on 16GB GPU
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-prop", type=float, default=0.8)
    args = parser.parse_args()

    seed_everything(args.seed)

    # TODO: Automate this process using tune.
    # If the train_dataset_size < 10000
    # rerun the training procedure 10 times with different random seeds.
    # After each run, output the test results.
    # Else, just run once.

    classifier = YelpConditionalSentenceGenerator(
        "bert-optimus-cased-snli-latent-768-beta-1",
        "gpt2-optimus-cased-snli-beta-1",
        val_dataset_size=args.val_dataset_size,
        train_dataset_size=args.train_dataset_size,
        batch_size=args.batch_size,
        train_prop=args.train_prop,
    )

    n_batches = args.train_dataset_size // args.batch_size
    _log_freq = min(args.log_freq, n_batches)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        val_check_interval=_log_freq,
        log_every_n_steps=_log_freq,
        accelerator="gpu",
        devices=[0],
        default_root_dir="./lightning_logs_generate",
    )
    trainer.fit(classifier)
