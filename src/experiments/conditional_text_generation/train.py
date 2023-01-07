import argparse

import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

from .conditional_text_generation import YelpConditionalSentenceGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--max-length", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-freq", type=int, default=100)
    args = parser.parse_args()

    seed_everything(args.seed)

    model = YelpConditionalSentenceGenerator.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        strict=False,
        max_length=args.max_length,
        batch_size=args.batch_size,
        val_dataset_size=50,
        use_beta_schedule=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        val_check_interval=args.log_freq,
        log_every_n_steps=args.log_freq,
        accelerator="gpu",
        devices=[0],
        default_root_dir="./lightning_logs_generate_default_base_model",
    )
    trainer.fit(model)
