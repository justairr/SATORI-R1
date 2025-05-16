cd VLMEvalKit

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc-per-node=4 run.py --data MMBench MMStar --model Qwen2.5-VL-3B-Instruct --verbose