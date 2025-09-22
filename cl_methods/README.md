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

### 2. Data and Model Preparation

-   1. Extract the provided test set from the full dataset. 

-   2. Place the splits within "data" folder as shown in the structure above. 

-   3. Load and save model using "cl_methods/model/init_model.py"

### 3. Training and Evaluating

To train, run: 

```
# Run training script with default parameters ("ewc.sh" or "lora.sh"):
sh scripts/ewc/ewc.sh
```

Key parameters in the training script:

-   `--model_name_or_path`: Path to the pretrained model
-   `--data_path`: Path to the training dataset
-   `--dataset_name`: Names of the datasets to train on
-   `--reg`: Regularization parameter (default: 0.5)
-   `--num_train_epochs`: Number of training epochs per task (default: 30)


## Evaluate
To evaluate on the two test sets (valence_shifting and valence stable) and calculate the perplexities, run: 

-   1. For EWC and ST: 
```
python eval.py --model_dir ./outputs/$model_dir --test_data_dir ./data

```

-   2. For LoRA: 
```
python eval_lora.py --model_dir ./outputs/$model_dir --test_data_dir ./data

```