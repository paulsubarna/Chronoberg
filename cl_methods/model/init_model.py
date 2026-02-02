import sys 
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers'])
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, AutoTokenizer
import os

# Model name and save directory
model_name = "EleutherAI/pythia-160m"
save_dir = "/app/src/Chronoberg/cl_methods/pythia_models"

os.makedirs(save_dir, exist_ok=True)

# Load the correct Pythia tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)

# Load Pythia configuration
config = GPTNeoXConfig.from_pretrained(model_name)
config.vocab_size = len(tokenizer)

# Initialize model from scratch
model = GPTNeoXForCausalLM(config)

# Ensure embeddings match tokenizer
model.resize_token_embeddings(len(tokenizer))

# Save
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Saved fresh Pythia-160m model and tokenizer to {save_dir}")