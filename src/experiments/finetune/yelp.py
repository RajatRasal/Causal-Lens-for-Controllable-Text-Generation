import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

from src.pretrained_optimus.base_yelp import YelpPreTrainedOptimus

if __name__ == "__main__":
    seed_everything(42)

    model = YelpPreTrainedOptimus(
        pretrained_latent_dim=768,
        pretrained_beta=1.0,
        pretrained_dataset="wiki",
        beta_cycle_len=1,
        max_length=64,
        batch_size=56,
        train_prop=0.9,
        train_dataset_size=450000,
        val_dataset_size=1000,
    )

    log_freq = 1000
    trainer = pl.Trainer(
        max_epochs=1,
        val_check_interval=log_freq,
        log_every_n_steps=log_freq,
        accelerator="gpu",
        devices=[0],
        default_root_dir="./lightning_logs_finetune",
    )
    trainer.fit(model)
