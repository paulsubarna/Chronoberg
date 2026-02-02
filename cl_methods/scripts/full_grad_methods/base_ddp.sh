now=$(date +"%m%d_%H%M%S")

model_name="pythia-160m"

iters=50000,50000,50000

OMP_NUM_THREADS=2 torchrun --nnodes=1 \
  --nproc_per_node=8 \
  /app/src/Chronoberg/cl_methods/training/main_ddp.py \
    --data_path /app/src/Chronoberg/cl_methods/dataset \
    --dataset_name 1750,1800,1850\
    --model_name_or_path "/app/src/Chronoberg/cl_methods/pythia_models" \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --num_train_iters $iters \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --print_loss \
    --CL_method base \
    --dtype bf16 \
    --beta2 0.99995 \
    --output_dir /app/src/Chronoberg/cl_methods/$model_name/base_$now