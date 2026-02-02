#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import sys
import subprocess
import time
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rouge'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rouge-score'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fuzzywuzzy'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'evaluate'])

from torch import nn


sys.dont_write_bytecode = True

import argparse
import os
import math
import sys
from tqdm import tqdm
from ivon import IVON_wprior
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)




sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset, DatasetWrapper
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

# # add flash attention
from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn
#
replace_llama_attn_with_flash_attn()
replace_bloom_attn_with_flash_attn()

from params import Method2Class, AllDatasetName


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='/app/src/Chronoberg/cl_methods/dataset',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name',
                        type=list_of_strings,
                        default='all',
                        help='Dataset to be used.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='./tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_iters",
                        type=list_of_strings,
                        default=None,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=100,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=3,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser.add_argument('--print_tiktok',
                        action='store_true',
                        help='Prints tiktok at each step.')
    
    parser.add_argument('--CL_method',
                default=None,
                help='continual learning method used')
    
    parser.add_argument('--reg',
                        default=0.0,
                        type=float,
                        help='regularization term used in continual learning')
    
    
    parser.add_argument('--lora_depth',
                        default=-1,
                        type=int,
                        help='max depth of lora layers, -1 means no limit')

    parser.add_argument('--dtype',
                        default="auto",
                        type=str,
                        help='max depth of lora layers, -1 means no limit')
    parser.add_argument('--ess',
                        default=1e10,
                        type=float,
                        help='ESS for IVON')
    parser.add_argument('--hess',
                        default=0.1,
                        type=float,
                        help='Hessian init for IVON')
    parser.add_argument('--beta2',
                        default=0.9999,
                        type=float,
                        help='Beta2 for IVON')

    args = parser.parse_args()


    return args

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return global_rank, local_rank, world_size

    
class LMBlockCollator:
    def __call__(self, batch):
        return {
            "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
            "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
        }

def main():
    args = parse_args()

    ## DDP setup 
    global_rank, local_rank, world_size = setup_ddp()
    gpu = local_rank   
    if local_rank == -1:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda", local_rank)

    args.global_rank = global_rank #torch.distributed.get_rank()
    args.local_rank = local_rank


    # set batch size
    #ds_config[
    #    'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    #ds_config[
    #    'train_batch_size'] = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps          

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    torch.distributed.barrier()
    torch.cuda.set_device(gpu)
    device = torch.device('cuda:{}'.format(gpu))

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # common for causal LM batching

    
    model = create_hf_model(
    args.model_name_or_path,
    tokenizer=tokenizer,
    dtype=args.dtype,                  # "bf16" or "fp16"
    local_rank=local_rank,
    disable_dropout=args.disable_dropout,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model = DDP(
    model,
    device_ids=[local_rank],
    #output_device=local_rank,
    find_unused_parameters=False,   # REQUIRED for GPT-NeoX / Pythia
    )

    if args.gradient_checkpointing:
        model.module.gradient_checkpointing_enable()

    
    # # replace SiLU with ReLU
    # replace_silu_with_relu(model)
    
    # print_rank_0(model, args.global_rank)
    
    # print blue color:
    print_rank_0(f"\033[34m***** Model:\n {model} *****\033[0m", args.global_rank)
    
    train_task_list = {}
    eval_task_list = {}
    test_task_list = {}


    if args.dataset_name[0] == "all":
        Datasets = AllDatasetName
    else:
        Datasets = args.dataset_name
    print("Datasets used: ", Datasets)
    for dataset in Datasets:
        dataset_path = os.path.join(args.data_path,dataset)
        try:
            train_dataset = torch.load(os.path.join(args.data_path, f'train_{dataset}.pt'), weights_only=False)
            #eval_dataset = torch.load(os.path.join(args.data_path, f'eval_{dataset}.pt'))
            test_dataset = torch.load(os.path.join(args.data_path, f'test_{dataset}.pt'), weights_only=False)
            print_rank_0(f"Loaded cached dataset for {dataset} from {args.data_path}", args.global_rank)
        except FileNotFoundError:
            print_rank_0(f"Creating dataset for {dataset}...", args.global_rank)
            # Prepare the data
            train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
                args.local_rank,
                dataset_path,
                args.data_output_path,
                args.seed,
                sample_ratio=1.0,
                distributed=True
            )

            t = time.time()
            train_dataset = DatasetWrapper(train_dataset, tokenizer, args.max_prompt_len + args.max_ans_len)
            #eval_dataset = DatasetWrapper(eval_dataset, tokenizer, args.max_prompt_len + args.max_ans_len)
            test_dataset = DatasetWrapper(test_dataset, tokenizer, args.max_prompt_len + args.max_ans_len)
            if args.global_rank == 0:
                torch.save(train_dataset, os.path.join(args.data_path, f'train_{dataset}.pt'))
                torch.save(test_dataset, os.path.join(args.data_path, f'test_{dataset}.pt'))
            print_rank_0(f"Time for tokenizing the dataset {dataset}: {time.time() - t} seconds.", args.global_rank)            

        # To ensure different process has different shuffling
        rng = torch.Generator()
        rng.manual_seed(args.seed + args.global_rank)
        train_dataloader = DataLoader(train_dataset,
                                    batch_size=args.per_device_train_batch_size,
                                    shuffle=True,
                                    generator=rng)
        #eval_dataloader = DataLoader(eval_dataset,
        #                            batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset,
                            batch_size=args.per_device_eval_batch_size)
        train_task_list[dataset] = train_dataloader
        #eval_task_list[dataset] = eval_dataloader
        test_task_list[dataset] = test_dataloader


    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                #  check output
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay
    )

    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps= 1e-8,
        betas=(0.9, 0.95),
    )
    '''
    optimizer = IVON_wprior(model.parameters(), 
                            lr=args.learning_rate, 
                            ess= args.ess, #len(train_loader.dataset), 
                            mc_samples=1,
                            hess_init=args.hess, 
                            beta2=args.beta2, 
                            weight_decay=args.weight_decay,
                            hess_approx='bonnet',
                            sync=True
                            #clip_radius=0.0001
                            ) 
    '''
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
    )
    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)
    
    def is_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except TypeError:
            return False
        
    # save args to output_dir:
    if args.output_dir and global_rank == 0:        # os.makedirs(args.output_dir, exist_ok=True)
        os.system(f"mkdir -p {args.output_dir}")
        # saving formated json file:
        # save all configs to args.tb_dir:
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            # write formatted json:
            import json
            # json.dump(vars(args), f, indent=4)
            # only saving JSON serializable properties:
            serializable_args = {k: v for k, v in vars(args).items() if is_serializable(v)}
            json.dump(serializable_args, f, indent=4)
            # copy ./model folder to tb_dir:
            os.system(f'cp -r -p model {args.output_dir}')
        
        

    # Initialize the global progress bar

    if args.CL_method in Method2Class.keys():
        CL_Trainer = Method2Class[args.CL_method](model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args, lr_scheduler)
        CL_Trainer.train_continual()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()