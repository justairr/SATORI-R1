
# 🚀 SATORI: Spatially Anchored Task Optimization with Reinforcement Learning

🔗 This is the **official implementation** of **SATORI**.

![📊 Overview](./img/page_1.png)

---

## 🛠️ Requirements

To install dependencies:

```bash
conda env create -f environment.yaml
````

We use [**ms-swift**](https://github.com/modelscope/ms-swift) to train the model. Install via:

```bash
pip install ms-swift -U
```

*or:*

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

---

## 🎯 Training

Run the following command to start training:

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

> **Note:**
>
> * `external_format` formats the data
> * `external_bbox_acc` computes bounding-box accuracy
> * `extern_vqa_acc` computes VQA accuracy
>   (See `plugin.py` for details) 🔍

---

## 📈 Evaluation

We provide an evaluation script in the `VLMEvalKit` directory. Make sure `VLMEvalKit` is in your working directory and update the dataset/model paths in `config.py`:

```text
# File: ./VLMEvalKit/vlmeval/config.py (line 417)
```

### 🏃‍♂️ Run Evaluation

```bash
cd VLMEvalKit
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc-per-node=4 run.py \
  --data MMBench MMStar \
  --model Qwen2.5-VL-3B-Instruct \
  --verbose
```

---

## 💾 Pretrained Models

Download here:

* 🔗 [SATORI-3B checkpoint](..)
* 🔗 [Qwen2.5-VL-3B-Instruct on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

---

## 📚 Dataset: VQA-Verify

We release the **VQA-Verify** dataset here: [link]() 🚀

---

## 🤝 Contributing

We welcome your **issues**, **PRs**, and **feedback**!
Feel free to open an issue or submit a pull request. 🙌


