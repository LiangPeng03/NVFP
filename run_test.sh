#!/bin/bash
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

MODEL="HuggingFaceTB/SmolLM2-135M"

$HOME/.conda/envs/awq/bin/python model_quant.py \
    --model_name_or_path=${MODEL} \
    --format=nvfp \
    --w_bits=4 \
    --a_bits=16 \
    --w_group_size=16 \
    --a_group_size=16 \
    --transform_class=hadamard \
    --w_observer=minmax \
    --quantization_order=default \
    --hadamard_group_size=64 \
    --dataset_name_or_path=c4 \
    --num_sequences=128 \
    --gptq \
    --rel_damp=0.01 \
    --sequence_length=2048 \
    --dtype=bfloat16 \
    --fuse_global_scale \
    --eval_perplexity \
    --amp
