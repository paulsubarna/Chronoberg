import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import Optional, Any

@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: bool = True                 
    max_prompt_len: Optional[int] = None
    max_ans_len: Optional[int] = None
    max_seq_len: Optional[int] = None 
    pad_to_multiple_of: Optional[int] = 1
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    inference: bool = False
    demonstrations: Optional[Any] = None
    task: str = None

    def __call__(self, batch):
        """
        Batch is a list of dicts with a 'text' field (from LocalTextDataset)
        """
        tokenized_batch = []
        sources = []  # store original text
        actual_max_len = 0

        # tokenize each example
        for instance in batch:
            text = instance['text']
            sources.append(text)  # keep original text
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                padding=False,
                add_special_tokens=True,
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            tokenized_batch.append(tokenized)

            if len(tokenized["input_ids"]) > actual_max_len:
                actual_max_len = len(tokenized["input_ids"])

        # round up to pad_to_multiple_of
        actual_pad_len = ((actual_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        # pad each example
        for example in tokenized_batch:
            pad_len = actual_pad_len - len(example["input_ids"])
            example["input_ids"] = [self.tokenizer.pad_token_id] * pad_len + example["input_ids"]
            example["attention_mask"] = [0] * pad_len + example["attention_mask"]
            example["labels"] = [self.label_pad_token_id] * pad_len + example["labels"]
            assert len(example["input_ids"]) == len(example["attention_mask"]) == len(example["labels"]) == actual_pad_len

        model_inputs = {
            "input_ids": torch.tensor([ex["input_ids"] for ex in tokenized_batch], dtype=torch.long),
            "attention_mask": torch.tensor([ex["attention_mask"] for ex in tokenized_batch], dtype=torch.long),
            "labels": torch.tensor([ex["labels"] for ex in tokenized_batch], dtype=torch.long),
            "sources": sources  # add original text for evaluation/logging
        }

        return model_inputs