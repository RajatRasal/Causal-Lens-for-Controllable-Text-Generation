import argparse

import torch
from lightning_lite.utilities.seed import seed_everything

from .yelp_conditional_text_generation2 import YelpConditionalSentenceGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond-label", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    seed_everything(args.seed)

    model = YelpConditionalSentenceGenerator.load_from_checkpoint(
        checkpoint_path="/home/ubuntu/Causal-Lens-for-Controllable-Text-Generation/lightning_logs_generate2/lightning_logs/version_2/checkpoints/epoch=0-step=8000.ckpt",  # noqa: E501
    ).eval()
    # model.cara.top_p = 1.0
    # model.cara.top_k = 0

    for _ in range(10):
        cond_labels = torch.tensor([args.cond_label]).long()
        input_seq_ids = torch.tensor([[101]])
        attention_mask = torch.tensor([[True]])
        pooled_hidden_fea = model.encoder(
            input_seq_ids, attention_mask=attention_mask
        )[1]
        latent_z = model.cara.linear(pooled_hidden_fea)
        label_emb = model.cara.label_embedding(cond_labels)
        past = latent_z + label_emb
        generated = model.conditional_generation(past)
        # generated = model.cara.sample_sequence_conditional_batch(
        #     past=past, context=model.cara.bos_token_id_list
        # )
        print(model.untokenise(generated.squeeze(0)))
