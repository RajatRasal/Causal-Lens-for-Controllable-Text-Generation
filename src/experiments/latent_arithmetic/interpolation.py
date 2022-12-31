import argparse

from lightning_lite.utilities.seed import seed_everything

from .pretrained_optimus.vae import PreTrainedOptimus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-model-name",
        type=str,
        default="bert-optimus-cased-snli-latent-768-beta-1",
    )
    parser.add_argument(
        "--decoder-model-name",
        type=str,
        default="gpt2-optimus-cased-snli-beta-1",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--sent-source",
        type=str,
        default="a yellow cat likes to chase a long string .",
    )
    parser.add_argument(
        "--sent-target",
        type=str,
        default="a yellow cat likes to chase a short string .",
    )
    parser.add_argument("--num-interpolation-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    optimus = PreTrainedOptimus(
        args.encoder_model_name, args.decoder_model_name
    ).eval()
    res = optimus.interpolate(
        args.sent_source, args.sent_target, args.num_interpolation_steps
    )
    for k, v in res.items():
        print(k, v)
