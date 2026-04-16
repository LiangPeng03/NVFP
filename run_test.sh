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
    --transform_class=identity \
    --w_observer=minmax \
    --quantization_order=default \
    --hadamard_group_size=128 \
    --dataset_name_or_path=c4 \
    --num_sequences=16 \
    --sequence_length=1024 \
    --dtype=bfloat16 \
    --eval_perplexity \
    --amp
