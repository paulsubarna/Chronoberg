#!/bin/bash

#get current time:
now=$(date +"%m%d_%H%M%S")

#get GPUs:
gpu_nodes="0,1"

# Use Pythia 1.4B model, or local model path
model_name="EleutherAI/pythia-1.4b" 

# 30 epochs for each task
epochs=30

# Task names and their corresponding datasets. The task and dataset names should match.
tasks=("1750") # ("1800" "1850" "1900" "1950")

# Start training on Task 1:
for i in ${!tasks[@]}; do
    task=${tasks[$i]}
    
    # Set the dataset path dynamically based on the task name
    data_path="./data/$task"  # each task has a folder with its dataset

    # Train on each task for 30 epochs:
    deepspeed --include=localhost:$gpu_nodes --master_port 25001 training/main.py  \
        --data_path $data_path \
        --dataset_name $task \
        --model_name_or_path ./PTM/$model_name \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --max_prompt_len 1024 \
        --max_ans_len 512 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --num_train_epochs 30 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 1234 \
        --zero_stage 2 \
        --deepspeed \
        --print_loss \
        --CL_method lora \
        --output_dir ./outputs/cl/$model_name/lora_$now/$task \
        --save_steps 1000  # Save after every 1000 steps or adjust as needed
    
    # Save model checkpoint after each task:
    model_path="./outputs/cl/$model_name/lora_$now/$task"
    
    #Save model checkpoint explicitly (optional):
    # cp -r $model_path ./PTM/$model_name  # Update this to fit your directory structure

done
