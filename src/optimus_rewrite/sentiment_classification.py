from torch.optim import Adam
from torch.utils.data import DataLoader

from .vae import PreTrainedOptimus


class YelpSentimentClassifier(PreTrainedOptimus):
    def __init__(
        self,
        bert_model_name: str,
        gpt2_model_name: str,
        latent_size: int = 32,
        max_length: int = 64,
        dataset_size: int = 10000,
    ):
        super().__init__(bert_model_name, gpt2_model_name, latent_size)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return Adam()

    def train_dataloader(self):
        # Filter sentences by max_length
        # Randomly sample self.dataset_size no .of sentences
        # Tokenise each sentence
        return DataLoader()

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()
