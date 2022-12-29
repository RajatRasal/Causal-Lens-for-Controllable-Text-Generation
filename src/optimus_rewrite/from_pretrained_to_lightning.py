import argparse
import os

import torch
from lightning_lite.utilities.seed import seed_everything

from .vae import PreTrainedOptimus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=508523)
    parser.add_argument("--latent-size", type=int, default=32)
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
    parser.add_argument(
        "--sent-input",
        type=str,
        default="a brown dog likes to eat long pasta .",
    )
    parser.add_argument("--num-interpolation-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    seed_everything(42)

    e = args.step
    output_encoder_dir = os.path.join(
        args.output_dir, "checkpoint-encoder-{}".format(e)
    )
    output_decoder_dir = os.path.join(
        args.output_dir, "checkpoint-decoder-{}".format(e)
    )
    output_full_dir = os.path.join(
        args.output_dir, "checkpoint-full-{}".format(e)
    )
    checkpoint = torch.load(
        os.path.join(output_full_dir, "training.bin"),
        map_location=torch.device("cpu"),
    )

    optimus = PreTrainedOptimus(
        output_encoder_dir, output_decoder_dir, latent_size=args.latent_size
    ).eval()
    res = optimus.interpolate(
        args.sent_source, args.sent_target, args.num_interpolation_steps
    )
    for k, v in res.items():
        print(k, v)

    print()

    res = optimus.analogy(
        args.sent_source,
        args.sent_target,
        args.sent_input,
        args.num_interpolation_steps,
    )
    print(res)

    print()

    res = optimus.reconstruct(args.sent_source)
    print(res)
