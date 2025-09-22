import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPTNeoXForCausalLM
import deepspeed

from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.my_peft import PeftModel
from model.lora import lora  # LoRA wrapper

# Config
MODEL_DIR = "Path to saved LoRA model" 
MAX_LENGTH = 512
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SETS = {
    "valence_shift": "./data/test_valence_shift.jsonl",
    "valence_stable": "./data/test_valence_stable.jsonl"
}

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = GPTNeoXForCausalLM.from_pretrained(MODEL_DIR)
# Wrap base model with LoRA weights
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

# DeepSpeed inference engine
ds_engine = deepspeed.init_inference(
    model,
    mp_size=1,                  
    dtype=torch.float16,        
    replace_method='auto',
    replace_with_kernel_inject=True
)
ds_engine.to(DEVICE)

# Load test datasets
test_dataloaders = {}
for name, path in TEST_SETS.items():
    dataset = create_prompt_dataset(path, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=DataCollator(tokenizer, max_length=MAX_LENGTH)
    )
    test_dataloaders[name] = dataloader

# Prepare CL wrapper for evaluation
args = type("", (), {})()  
args.local_rank = -1
args.global_rank = 0

cl_model = lora(
    model=ds_engine,
    tokenizer=tokenizer,
    optimizer=None,
    train_task_list={},
    eval_task_list=test_dataloaders,
    test_task_list={},
    args=args
)

# Evaluate perplexity on each test set
for name, dataloader in test_dataloaders.items():
    ppl = cl_model.perplexity_evaluation(dataloader, DEVICE)
    print(f"Perplexity on {name} test set: {ppl:.2f}")
