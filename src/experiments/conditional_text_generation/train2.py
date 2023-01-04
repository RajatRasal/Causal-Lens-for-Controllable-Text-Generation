import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

from .yelp_conditional_text_generation2 import YelpConditionalSentenceGenerator

if __name__ == "__main__":
    seed_everything(42)

    model = YelpConditionalSentenceGenerator.load_from_checkpoint(
        checkpoint_path="/home/ubuntu/Causal-Lens-for-Controllable-Text-Generation/lightning_logs_finetune/lightning_logs/version_21/checkpoints/epoch=0-step=8000.ckpt",  # noqa: E501
        strict=False,
        val_dataset_size=50,
        use_beta_schedule=False,
    )
    log_freq = 100
    trainer = pl.Trainer(
        max_epochs=1,
        val_check_interval=log_freq,
        log_every_n_steps=log_freq,
        accelerator="gpu",
        devices=[0],
        default_root_dir="./lightning_logs_generate2",
    )
    trainer.fit(model)
