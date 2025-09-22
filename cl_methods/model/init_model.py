from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, AutoTokenizer
import torch

# Architecture 
model_name = "EleutherAI/pythia-1.4b"
# Folder to save model and tokenizer  
save_dir = "/PTM"                       

# Load GPT-NeoX tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Load Pythia-1.4B config
config = GPTNeoXConfig.from_pretrained(model_name)
config.vocab_size = len(tokenizer)

# Initialize model from scratch
model = GPTNeoXForCausalLM(config)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Saved fresh Pythia-1.4B model and tokenizer to {save_dir}")

"""
### Sanity check
# Reload to check consistency
model_reload = GPTNeoXForCausalLM.from_pretrained(save_dir)
tokenizer_reload = AutoTokenizer.from_pretrained(save_dir)

vocab_size_tok = len(tokenizer_reload)
vocab_size_model = model_reload.get_input_embeddings().weight.shape[0]

print(f"Tokenizer vocab size: {vocab_size_tok}")
print(f"Model embedding size: {vocab_size_model}")

if vocab_size_tok == vocab_size_model:
    print("tokenizer and model vocab sizes match.")
else:
    print("Mismatch detected! Set config.vocab_size = len(tokenizer)")
"""
