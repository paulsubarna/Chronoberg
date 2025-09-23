#  Sequential training of LLMs on ChronoBerg

This folder contains the implementation for EWC, LoRA and Sequential Training.

## Code Structure

```
.
├── data/                    # Dataset directory
│   ├── dataset_name_1/      # name of the task (eg.: 1750)
│   |    └── train.json/    
│   |    └── eval.json/     
│   |    └── test.json/     
│   ├── dataset_name_2/      # name of the second task (eg.: 1800)
├── model/                   # Model implementations
│   ├── ewc/                 
│   │   └── ewc.py           # EWC implementation
│   ├── lora/                 
│   │   └── lora.py          # LoRA implementation
│   └── base_model.py        # Base model implementation
├── scripts/                 # Training scripts 
├── training/                # Training related code
│   ├── main.py              # Main training script
│   └── params.py            # Training parameters
├── utils/                   # Utility functions
│   ├── data/                # Data processing utilities
│   ├── my_peft/             # Custom PEFT implementations
├── eval/                    # Evaluate on valence stable and valence shifting test sets
│   ├── eval_lora.py         # Calculate perplexity for LoRA 
│   ├── eval.py              # Calculate perplexity for EWC and ST
```

## Requirements

The main dependencies are listed below. For a complete list, see `requirements.txt`:

```
accelerate==1.0.1
bitsandbytes==0.46.1
deepspeed==0.15.3+cu124torch2.4
torch==2.4.1
torchvision==0.19.1
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
[TODO] : pip install -r requirements.txt
```

