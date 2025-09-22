###############################################################################################
#python evaluate_base_model.py --model_dir ./outputs/$model --test_data_dir ./data
###############################################################################################

import torch
from torch.utils.data import DataLoader
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from model.base_model import CL_Base_Model
from transformers import AutoTokenizer, GPTNeoXForCausalLM
import deepspeed
import argparse
import os

# Args parser
parser = argparse.ArgumentParser(description="Evaluate base and EWC models on ChronoBerg test sets")
parser.add_argument("--model_dir", type=str, required=True, help="Path to saved base model")
parser.add_argument("--test_data_dir", type=str, default="./data", help="Path to test JSONL datasets")
parser.add_argument("--max_length", type=int, default=512, help="Max token length for prompts")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
args = parser.parse_args()

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = GPTNeoXForCausalLM.from_pretrained(args.model_dir)
model.eval()

# Initialize inference

ds_engine = deepspeed.init_inference(
    model,
    mp_size=1,                  
    dtype=torch.float16,      
    replace_method='auto',
    replace_with_kernel_inject=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds_engine.to(device)

# Load test datasets
test_sets = {
    "valence_shift": os.path.join(args.test_data_dir, "test_valence_shift.jsonl"),
    "valence_stable": os.path.join(args.test_data_dir, "test_valence_stable.jsonl")
}

test_dataloaders = {}
for name, path in test_sets.items():
    dataset = create_prompt_dataset(path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollator(tokenizer, max_length=args.max_length)
    )
    test_dataloaders[name] = dataloader


# Wrap in CL_Base_Model
class Args:
    local_rank = -1
    global_rank = 0

cl_model = CL_Base_Model(
    model=ds_engine,
    tokenizer=tokenizer,
    optimizer=None,
    train_task_list={},
    eval_task_list=test_dataloaders,
    test_task_list={},
    args=Args()
)

# Calculate perplexity
print("Starting evaluation...\n")
for name, dataloader in test_dataloaders.items():
    ppl = cl_model.perplexity_evaluation(dataloader, device)
    print(f"Perplexity on {name} test set: {ppl:.2f}")
