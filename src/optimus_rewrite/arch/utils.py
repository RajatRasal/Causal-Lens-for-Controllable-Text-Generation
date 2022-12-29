import json
import logging
import math
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)


def safe_log(z):
    return torch.log(z + 1e-7)


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(
            torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim)
        )
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def generate_grid(zmin, zmax, dz, device, ndim=2):
    """generate a 1- or 2-dimensional grid
    Returns: Tensor, int
        Tensor: The grid tensor with shape (k^2, 2),
            where k=(zmax - zmin)/dz
        int: k
    """

    if ndim == 2:
        x = torch.arange(zmin, zmax, dz)
        k = x.size(0)

        x1 = x.unsqueeze(1).repeat(1, k).view(-1)
        x2 = x.repeat(k)

        return (
            torch.cat((x1.unsqueeze(-1), x2.unsqueeze(-1)), dim=-1).to(device),
            k,
        )

    elif ndim == 1:
        return torch.arange(zmin, zmax, dz).unsqueeze(1).to(device)


class BucketSampler(Sampler):
    def __init__(
        self, lens, bucket_size, batch_size, droplast=False, shuffle=True
    ):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [
            sorted(
                ids[i : i + self._bucket_size],  # noqa: E203
                key=lambda i: self._lens[i],
                reverse=True,
            )
            for i in range(0, len(ids), self._bucket_size)
        ]
        batches = [
            bucket[i : i + self._batch_size]  # noqa: E203
            for bucket in buckets
            for i in range(0, len(bucket), self._batch_size)
        ]
        if self._droplast:
            batches = [
                batch for batch in batches if len(batch) == self._batch_size
            ]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = [self._bucket_size] * (
            len(self._lens) // self._bucket_size
        ) + [len(self._lens) % self._bucket_size]
        if self._droplast:
            return sum(s // self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s / self._batch_size) for s in bucket_sizes)


class TokenDataset(Dataset):
    def __init__(
        self,
        tokenizers,
        args,
        file_path="train",
        text_split_mode="natural",
        block_size=512,
    ):

        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, f"cached_lm_gpt_bert_{block_size}_{filename[:-4]}.json"
        )

        self.examples = []
        self.tokenizers = tokenizers

        # Bert tokenizer special tokens
        self.bert_pad_token = tokenizers[0].convert_tokens_to_ids(
            [tokenizers[0].pad_token]
        )[0]

        # GPT-2 tokenizer special tokens
        self.gpt2_pad_token = tokenizers[1].convert_tokens_to_ids(
            [tokenizers[1].pad_token]
        )[0]
        self.gpt2_bos_token = tokenizers[1].convert_tokens_to_ids(
            [tokenizers[1].bos_token]
        )[0]
        self.gpt2_eos_token = tokenizers[1].convert_tokens_to_ids(
            [tokenizers[1].eos_token]
        )[0]

        global bert_pad_token
        global gpt2_pad_token
        bert_pad_token = self.bert_pad_token
        gpt2_pad_token = self.gpt2_pad_token

        if args.dataset == "Yelp":
            label_on = True
        else:
            label_on = False

        if os.path.exists(cached_features_file):
            logger.info(
                "Loading features from cached file %s", cached_features_file
            )
            with open(cached_features_file) as handle:
                self.examples = json.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            dropped, count = self._read_corpus_natural_split(
                fname=file_path,
                label=label_on,
                max_length=block_size,
                block_size=block_size,
                args=args,
            )

            logger.info("The number of dropped sentences is %d", dropped)
            logger.info("The number of processed sentences is %d", count)

            # Note that we are loosing the last truncated example here for the
            # sake of simplicity (no padding). If your dataset is small, first
            # you should loook for a bigger one :-) and second you can change
            # this behavior by adding (model specific) padding.

            logger.info(
                "Saving features into cached file %s", cached_features_file
            )
            if args.use_philly:
                save_solid = False
                while not save_solid:
                    try:
                        with open(cached_features_file, "w") as handle:
                            json.dump(self.examples, handle)
                    except Exception:
                        pass
            else:
                with open(cached_features_file, "w") as handle:
                    json.dump(self.examples, handle)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def collate(examples):
        # Convert to Tensors and build dataset
        input_ids_bert = pad_sequence(
            [
                torch.tensor(f["bert_token"], dtype=torch.long)
                for f in examples
            ],  # noqa: E501
            batch_first=True,
            padding_value=bert_pad_token,
        )
        input_ids_gpt = pad_sequence(
            [
                torch.tensor(f["gpt2_token"], dtype=torch.long)
                for f in examples
            ],  # noqa: E501
            batch_first=True,
            padding_value=gpt2_pad_token,
        )
        token_lengths = torch.tensor(
            [
                [f["bert_token_length"], f["gpt2_token_length"]]
                for f in examples
            ],
            dtype=torch.long,
        )

        return (input_ids_bert, input_ids_gpt, token_lengths)

    def _read_corpus_natural_split(
        self, fname, label, max_length, block_size, args
    ):
        labels = [] if label else None
        dropped = 0
        count = 0

        with open(fname) as fin:
            for line in fin:
                if label:
                    split_line = line.split("\t")
                    lb = split_line[0]
                    split_line_text = split_line[1]
                else:
                    split_line_text = line
                    split_line_text = split_line_text.strip()

                if len(split_line_text.split()) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line_text.split()) > max_length:
                        dropped += 1
                        continue

                if label:
                    labels.append(lb)

                tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(
                    self.tokenizers[0].tokenize(split_line_text)
                )
                tokenized_text0 = self.tokenizers[
                    0
                ].add_special_tokens_single_sentence(tokenized_text0)
                tokenized_text0_length = len(tokenized_text0)

                tokenized_text1 = self.tokenizers[1].convert_tokens_to_ids(
                    self.tokenizers[1].tokenize(split_line_text)
                )
                tokenized_text1 = self.tokenizers[
                    1
                ].add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1 = (
                    [self.gpt2_bos_token]
                    + tokenized_text1
                    + [self.gpt2_eos_token]
                )
                tokenized_text1_length = len(tokenized_text1)

                example = {
                    "bert_token": tokenized_text0,
                    "bert_token_length": tokenized_text0_length,
                    "gpt2_token": tokenized_text1,
                    "gpt2_token_length": tokenized_text1_length,
                }
                self.examples.append(example)
                count += 1

        return dropped, count


class BucketingDataLoader:
    def __init__(
        self,
        file_path,
        batch_size,
        max_seq_length,
        tokenizer,
        args,
        bucket=100,
        shuffle=True,
    ):

        self.dataset = TokenDataset(
            tokenizer, args, file_path, block_size=args.block_size
        )
        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples // batch_size
        self.example_lengths = [
            example["bert_token_length"] for example in self.dataset.examples
        ]

    def __iter__(self):
        sampler = BucketSampler(
            self.example_lengths,
            self.bucket_size,
            self.batch_size,
            droplast=True,
            shuffle=self.shuffle,
        )
        loader = DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=TokenDataset.collate,
        )
        yield from loader

    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass
