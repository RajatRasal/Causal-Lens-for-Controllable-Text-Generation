import argparse

import torch
from lightning_lite.utilities.seed import seed_everything

from .conditional_text_generation import YelpConditionalSentenceGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    model = YelpConditionalSentenceGenerator.load_from_checkpoint(
        checkpoint_path="/home/ubuntu/Causal-Lens-for-Controllable-Text-Generation/lightning_logs_generate2/lightning_logs/version_7/checkpoints/epoch=29-step=82076.ckpt",  # noqa: E501
    ).eval()
    dataloader = model.test_dataloader()
    # model.cara.top_p = 1.0
    # model.cara.top_k = 0

    def _gen_helper(tokens, cond_labels):
        label_emb = model.cara.label_embedding(cond_labels)
        tokens = tokens.unsqueeze(0)
        mask = (tokens > 0).float()
        pooled_hidden_fea = model.encoder(tokens, attention_mask=mask)[1]
        latent_z = model.cara.linear(pooled_hidden_fea)
        past = latent_z + label_emb
        generated = model.conditional_generation(past)
        return model.untokenise(generated.squeeze(0))

    for _batch in dataloader:
        batch = _batch.tokens_batch
        for sent, label, tokens in zip(
            batch.sentences, _batch.labels, batch.enc_tokens_batch
        ):
            print(label.item(), "-", sent)
            cond_0 = torch.tensor([0]).long()
            cond_05 = torch.tensor([0.5]).long()
            cond_1 = torch.tensor([1]).long()
            print("0", _gen_helper(tokens, cond_0))
            print("05", _gen_helper(tokens, cond_05))
            print("1", _gen_helper(tokens, cond_1))
            print()
        break
