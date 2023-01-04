import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

from src.pretrained_optimus.base_yelp import YelpPreTrainedOptimus

if __name__ == "__main__":
    seed_everything(42)

    model = YelpPreTrainedOptimus(
        bert_model_name="bert-optimus-cased-snli-latent-768-beta-1",
        gpt2_model_name="gpt2-optimus-cased-snli-beta-1",
        beta_cycle_len=1,
        max_length=64,
        batch_size=56,
        train_prop=0.9,
        train_dataset_size=450000,
        val_dataset_size=1000,
    )

    log_freq = 1000
    trainer = pl.Trainer(
        max_epochs=1,  # args.max_epochs,
        val_check_interval=log_freq,
        log_every_n_steps=log_freq,
        accelerator="gpu",
        devices=[0],
        default_root_dir="./lightning_logs_finetune",
    )
    trainer.fit(model)
