import argparse
from collections import defaultdict
from typing import Dict

from lightning_lite.utilities.seed import seed_everything

from .pretrained_optimus.vae import PreTrainedOptimus


def interpolate(
    model: PreTrainedOptimus,
    source_sent: str,
    target_sent: str,
    steps: int = 10,
) -> Dict[int, str]:
    _, z1, _ = model.encode(model.tokenise([source_sent])[0].unsqueeze(0))
    _, z2, _ = model.encode(model.tokenise([target_sent])[0].unsqueeze(0))

    results = defaultdict(str)
    for step in range(steps + 1):
        z = z1 + (z2 - z1) * step * 1.0 / steps
        tokens = model.conditional_generation(z)
        text = model.untokenise(tokens.squeeze(0))
        results[step] = text

    return results


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
    res = interpolate(
        optimus,
        args.sent_source,
        args.sent_target,
        args.num_interpolation_steps,
    )
    for k, v in res.items():
        print(k, v)
