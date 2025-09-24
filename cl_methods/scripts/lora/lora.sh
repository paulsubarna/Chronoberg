# This is the script to run the tran and evaluate EWC method on the TRACE continual learning benchmark.
#get current time:
now=$(date +"%m%d_%H%M%S")
#get GPUs:
gpu_nodes="0"

model_name="pythia-1.4b"

#set all datasets and epochs for single-interval baselines.
epochs=2,2
#epochs=5,3,7,5,3,5,5,7

# Train:
deepspeed --include=localhost:$gpu_nodes --master_port 25000 training/main.py  \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name 1750,1800 \
    --model_name_or_path ./PTM/$model_name \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_train_epochs $epochs \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 100 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method lora \
    --output_dir ./outputs_LLM-CL/cl/$model_name/lora_$now
