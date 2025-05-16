## Training

We use [swift](https://github.com/modelscope/ms-swift) to train the model.
To install, please run:
```bash
pip install ms-swift -U
```
or:
```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```
For training, please run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=401408 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins ./plugin.py \
    --reward_funcs external_format external_bbox_acc extern_vqa_acc \
    --train_type full \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset DATASET_PATH \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --system './prompt.txt' \
    --log_completions true
```

> ORM functions are stated in the `plugin.py` file. We use the `external_format` function to format the data, `external_bbox_acc` to calculate the bounding box accuracy, and `extern_vqa_acc` to calculate the VQA accuracy. The `external_format` function is used to format the data for training. The `external_bbox_acc` function is used to calculate the bounding box accuracy. 

## Evaluation
This repository provides a script to evaluate the SATORI model on the SATORI dataset. The evaluation script is located in the `VLMEvalKit` directory. To run the evaluation, you need to have the `VLMEvalKit` directory in your working directory.
You should modify the `config.py` file to set the correct path for the dataset and the model.
config.py path: ./VLMEvalKit/vlmeval/config.py, line 417

### Evaluation Script
```bash
cd VLMEvalKit
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc-per-node=4 run.py --data MMBench MMStar --model Qwen2.5-VL-3B-Instruct --verbose
```

## Dataset:VQA-Verify
We release the VQA-Verify dataset in [link]().