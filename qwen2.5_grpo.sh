MAX_PIXELS=401408 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins ./plugin.py \
    --reward_funcs external_format external_bbox_acc extern_vqa_acc external_caption_acc \
    --use_vllm false \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.6 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset <DATA_PATH> \
    --max_length 2048 \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir output/GRPO \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --system './prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --vllm_max_model_len 1024 \
    --num_iterations 1 \
    --num_infer_workers 1 \
    --async_generate false \
    --beta 0.001 \