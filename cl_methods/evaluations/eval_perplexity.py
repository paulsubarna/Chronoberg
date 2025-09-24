#########################################################################################
# python eval.py \
#  --model_dir ./outputs_LLM-CL/cl/pythia-1.4b/lora_0924_111906/0 \
#  --stable_path ./data/valence_stable.json \
#  --shift_path ./data/valence_shift.json \
#  --batch_size 1
#########################################################################################

import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import argparse
from tqdm import tqdm
import json

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.texts = []
        with open(file_path, "r") as f:
            if file_path.endswith(".jsonl"):
                self.texts = [json.loads(line)["text"].strip() for line in f]
            elif file_path.endswith(".json"):
                data = json.load(f)
                if isinstance(data, dict):
                    # e.g. {"0.75": [...], "0.5": [...]}
                    for _, samples in data.items():
                        for text in samples:
                            clean = text.strip().replace("\n", " ")
                            if clean: 
                                self.texts.append(clean)
                elif isinstance(data, list):
                    # e.g. [{"text": "..."}]
                    for d in data:
                        clean = d.get("text", "").strip().replace("\n", " ")
                        if clean:
                            self.texts.append(clean)
            else:  # plain txt
                self.texts = [line.strip() for line in f if line.strip()]

        self.examples = self.tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=block_size,
            return_tensors="pt"
        )["input_ids"]

    def __len__(self):
        return self.examples.size(0)

    def __getitem__(self, idx):
        return self.examples[idx]

# ---- Perplexity Evaluation ----
def evaluate_perplexity(model, dataloader, device, tokenizer, desc="Evaluating"):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            batch = batch.to(device)
            outputs = model(batch)
            logits = outputs.logits  # [B, T, V]
            loss = loss_fct(logits.view(-1, logits.size(-1)), batch.view(-1))
            total_loss += loss.item()
            total_tokens += (batch != tokenizer.pad_token_id).sum().item()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

# ---- Main ----
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.to(device)

    # Datasets
    stable_dataset = TextDataset(args.stable_path, tokenizer)
    shift_dataset = TextDataset(args.shift_path, tokenizer)

    stable_loader = DataLoader(stable_dataset, batch_size=args.batch_size)
    shift_loader = DataLoader(shift_dataset, batch_size=args.batch_size)

    # Evaluate
    stable_ppl = evaluate_perplexity(model, stable_loader, device, tokenizer, desc="Stable set")
    shift_ppl  = evaluate_perplexity(model, shift_loader, device, tokenizer, desc="Shift set")

    print(f"\n=== Perplexity Results ===")
    print(f"Valence-stable set:   {stable_ppl:.2f}")
    print(f"Valence-shifting set: {shift_ppl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to saved model checkpoint dir")
    parser.add_argument("--stable_path", type=str, required=True,
                        help="Path to valence-stable test set (json/jsonl/txt)")
    parser.add_argument("--shift_path", type=str, required=True,
                        help="Path to valence-shifting test set (json/jsonl/txt)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=5,
                        help="Limit number of samples for debugging")
    args = parser.parse_args()
    main(args)