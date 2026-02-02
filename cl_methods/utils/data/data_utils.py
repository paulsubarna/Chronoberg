# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""

import os
from typing import List, Literal, Optional, TypedDict
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import torch.nn.functional as F
import numpy as np
import hashlib
from datasets import load_dataset

from tqdm import tqdm

class LocalTextDataset:
    """
    Handles a dataset for next-token prediction.
    Expects train.json, eval.json, test.json in `data_dir`.
    Each JSON file should be a list of objects with a 'text' field.
    """
    def __init__(self, data_dir):
        assert os.path.exists(data_dir), f"Data folder not found: {data_dir}"
        self.data_dir = data_dir
        self.raw_datasets = load_dataset("json", data_files={
            "train": f"{data_dir}/train.json",
            #"eval": f"{data_dir}/eval.json",
            "test": f"{data_dir}/test.json",
        })

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["eval"]

    def get_test_data(self):
        return self.raw_datasets["test"]

    # for compatibility
    def get_prompt(self, sample):
        return sample["text"]

    def get_answer(self, sample):
        return ""


def get_raw_dataset(dataset_name, output_path, seed, local_rank, for_backbone=False):
    """
    Return the raw dataset object.
    Supports RLHF datasets or local text dataset folder.
    """
    from . import raw_datasets
    if "Anthropic/hh-rlhf" in dataset_name:
        return raw_datasets.AnthropichhrlhfDataset(output_path, seed, local_rank, dataset_name)
    else:
        return LocalTextDataset(dataset_name)


def create_dataset(local_rank, dataset_name, output_path,
                   seed, for_backbone=False, sample_ratio=None):
    """
    Returns HuggingFace dataset objects for train/eval/test.
    """
    eval_dataset = None  # to avoid error
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank, for_backbone=for_backbone)

    train_dataset = raw_dataset.get_train_data()
    #eval_dataset = raw_dataset.get_eval_data()
    test_dataset = raw_dataset.get_test_data()

    # Optionally subsample
    if sample_ratio is not None:
        train_dataset = train_dataset.shuffle(seed=seed).select(range(int(len(train_dataset) * sample_ratio)))
        #eval_dataset = eval_dataset.shuffle(seed=seed).select(range(int(len(eval_dataset) * sample_ratio)))
        test_dataset = test_dataset.shuffle(seed=seed).select(range(int(len(test_dataset) * sample_ratio)))

    return train_dataset, eval_dataset, test_dataset








def make_lm_blocks(hf_ds, tokenizer, seq_len=1536, num_proc=4):
    # 1) tokenize each document; add EOS after each doc
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "Tokenizer has no eos_token_id"

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            add_special_tokens=False,
            truncation=False,
        )
        # append EOS between years
        out["input_ids"] = [ids + [eos_id] for ids in out["input_ids"]]
        out["attention_mask"] = [mask + [1] for mask in out["attention_mask"]]
        return out

    tokenized = hf_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=hf_ds.column_names,  # remove "text"
        num_proc=num_proc,
        desc="Tokenizing",
    )

    # 2) concatenate and split into fixed blocks
    def group_texts(examples):
        # concatenate lists of lists -> one long list
        input_ids = sum(examples["input_ids"], [])
        # (attention_mask is all 1s here, but keep it for completeness)
        # attn = sum(examples["attention_mask"], [])

        total_len = len(input_ids)
        total_len = (total_len // seq_len) * seq_len  # drop remainder (standard)

        blocks = [input_ids[i:i+seq_len] for i in range(0, total_len, seq_len)]

        return {
            "input_ids": blocks,
            "attention_mask": [[1]*seq_len for _ in blocks],
            "labels": [b.copy() for b in blocks],
        }

    lm_ds = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Grouping into {seq_len}-token blocks",
    )
    return lm_ds


def create_prompt_dataset(local_rank,
                          data_path,
                          output_path,
                          seed,
                          reload=False,
                          for_backbone=False,
                          distributed=True,
                          sample_ratio=None):
    """
    Creates and caches the next-token prediction dataset.
    """
    os.makedirs(output_path, exist_ok=True)
    fname = f"{data_path}_seed{seed}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest()

    train_fname = f"{output_path}/traindata_{fname}.pt"
    #eval_fname = f"{output_path}/evaldata_{fname}.pt"
    test_fname = f"{output_path}/testdata_{fname}.pt"

    #if local_rank <= 0:
    train_dataset, eval_dataset, test_dataset = create_dataset(
            local_rank, data_path, output_path, seed, for_backbone=for_backbone, sample_ratio=sample_ratio
    )

        #torch.save(train_dataset, train_fname)
        #torch.save(eval_dataset, eval_fname)
        #torch.save(test_dataset, test_fname)

    if distributed:
        torch.distributed.barrier()

    return train_dataset, eval_dataset, test_dataset # torch.load(train_fname), None, torch.load(test_fname)


class DatasetWrapper(Dataset):
    def __init__(self, doc_dataset, tokenizer, block_size):
        print("Tokenizing dataset...")
        texts = [doc['text'] for doc in doc_dataset]
        btok = tokenizer.backend_tokenizer
        encs = btok.encode_batch(texts, add_special_tokens=False)
        self._tokenized_docs = [torch.tensor(enc.ids) for enc in encs]
        print(f"Total #tokens: {sum(x.numel() for x in self._tokenized_docs)}")
        self._len = sum(len(x) - block_size + 1 for x in self._tokenized_docs)
        print(f"Total #samples: {self._len}")
        self._block_size = block_size

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        for tokens in self._tokenized_docs:
            n_windows = tokens.numel() - self._block_size + 1
            if idx < n_windows:
                return {
                    "input_ids": tokens[idx:idx + self._block_size],
                    "labels": tokens[idx:idx + self._block_size],
                    "sources": -1
                }
            idx -= n_windows
        raise IndexError("Index out of range")