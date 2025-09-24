#get current time:
now=$(date +"%m%d_%H%M%S")
#get GPUs:
#gpu_nodes="0,1"
gpu_nodes="0"

model_name="pythia-1.4b"

epochs=5
#epochs=2,1,3,2,1,2,2,3
#epochs=5,3,7,5,3,5,5,7

# Train:
deepspeed --include=localhost:$gpu_nodes --master_port 25000 training/main.py  \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name 1750 \
    --model_name_or_path ./PTM/$model_name \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --num_train_epochs $epochs \
    --gradient_accumulation_steps 5 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method base \
    --output_dir ./outputs_LLM-CL/cl/$model_name/base_$now
