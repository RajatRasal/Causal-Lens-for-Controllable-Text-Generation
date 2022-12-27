import argparse
import logging
from typing import List

from lightning_lite.utilities.seed import seed_everything

from src.optimus.vae import BertGPT2VAE, load_bert_gpt2_vae


def interpolate(
    model: BertGPT2VAE, a: str, b: str, samples: int = 1
) -> List[List[str]]:
    z_start, _, _ = model.encode(
        model.tokeniser_encoder.encode(
            a, max_length=64, return_tensors="pt", truncation=True
        )
    )
    z_end, _, _ = model.encode(
        model.tokeniser_encoder.encode(
            b, max_length=64, return_tensors="pt", truncation=True
        )
    )

    results = [[a]]
    for i in range(1, 10, 1):
        tau = i / 10
        z_tau = z_start * (1 - tau) + z_end * tau
        sent = model.conditional_generation(
            z_tau, max_length=64, num_return_sequences=samples
        )
        results.append(sent)
    results.append([b])

    return results


def sentence_arithmetic(
    model: BertGPT2VAE, a: str, b: str, c: str, samples: int = 1
) -> List[str]:
    z_a, _, _ = model.encode(
        model.tokeniser_encoder.encode(
            a, max_length=64, return_tensors="pt", truncation=True
        )
    )
    z_b, _, _ = model.encode(
        model.tokeniser_encoder.encode(
            b, max_length=64, return_tensors="pt", truncation=True
        )
    )
    z_c, _, _ = model.encode(
        model.tokeniser_encoder.encode(
            c, max_length=64, return_tensors="pt", truncation=True
        )
    )

    z_d = z_b - z_a + z_c

    sent = model.conditional_generation(
        z_d, max_length=64, num_return_sequences=samples
    )

    return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--checkpoint-path", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["interpolate", "arithmetic"],
        default="interpolate",
    )
    parser.add_argument("-a", "--sent-a", type=str, required=True)
    parser.add_argument("-b", "--sent-b", type=str, required=True)
    parser.add_argument("-c", "--sent-c", type=str)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)

    if args.mode == "arithmetic" and not args.sent_c:
        raise Exception("Sentence C is required for arithmetic mode.")

    seed_everything(args.seed)

    model = load_bert_gpt2_vae(args.checkpoint_path).eval()

    if args.mode == "arithmetic":
        sents = sentence_arithmetic(
            model, args.sent_a, args.sent_b, args.sent_c
        )
        print(sents)
    elif args.mode == "interpolate":
        results = interpolate(model, args.sent_a, args.sent_b)
        for i, row in enumerate(results):
            print(f"{i}: {row}")
    else:
        raise NotImplementedError(f"{args.mode} does not exist")
