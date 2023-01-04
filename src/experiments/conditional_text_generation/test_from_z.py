import argparse

import torch
from lightning_lite.utilities.seed import seed_everything

from .yelp_conditional_text_generation import YelpConditionalSentenceGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond-label", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    seed_everything(args.seed)

    model = YelpConditionalSentenceGenerator.load_from_checkpoint(
        checkpoint_path="/home/ubuntu/Causal-Lens-for-Controllable-Text-Generation/lightning_logs_generate2/lightning_logs/version_2/checkpoints/epoch=0-step=8000.ckpt",  # noqa: E501
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
            cond_label = torch.tensor([label]).long()
            cond_label_flip = torch.tensor([not label]).long()
            print("Same:", _gen_helper(tokens, cond_label))
            print("Flip:", _gen_helper(tokens, cond_label_flip))
            print()
        break
    # generated = model.cara.sample_sequence_conditional_batch(
    #     past=past, context=model.cara.bos_token_id_list
    # )
    # print(model.untokenise(generated.squeeze(0)))
