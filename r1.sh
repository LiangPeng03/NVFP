#!/bin/bash

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TQDM_DISABLE=1

MODEL1="HuggingFaceTB/SmolLM2-135M"
FORMAT="mxfp"

ALPHAS=(0 0.25 0.5 0.75 1)
MEAN_PENALTIES=(1)

RESULTS_FILE="grid_search_results.csv"
echo "timestamp,model,format,alpha,alpha_mean_penalty,wikitext_ppl,c4_ppl,status" > $RESULTS_FILE

TOTAL=$((${#ALPHAS[@]} * ${#MEAN_PENALTIES[@]}))
COUNT=0

for alpha in "${ALPHAS[@]}"; do
    for penalty in "${MEAN_PENALTIES[@]}"; do
        COUNT=$((COUNT + 1))
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        
        echo "========================================"
        echo "[$COUNT/$TOTAL] alpha=$alpha, alpha_mean_penalty=$penalty"
        echo "========================================"
        
        OUTPUT_DIR="experiments/alpha${alpha}_penalty${penalty}_${TIMESTAMP}"
        mkdir -p $OUTPUT_DIR
        
        # 只保留参数设置到 log
        {
            echo "timestamp=$TIMESTAMP"
            echo "model=$MODEL1"
            echo "format=$FORMAT"
            echo "alpha=$alpha"
            echo "alpha_mean_penalty=$penalty"
            echo "w_bits=4"
            echo "a_bits=4"
            echo "w_group_size=32"
            echo "a_group_size=32"
            echo "transform_class=hadamard"
            echo "w_observer=mse"
            echo "quantization_order=activation"
            echo "hadamard_group_size=64"
            echo "dataset_name_or_path=c4"
            echo "num_sequences=128"
            echo "rel_damp=0.01"
            echo "sequence_length=2048"
            echo "dtype=bfloat16"
        } > ${OUTPUT_DIR}/log.txt
        
        # 运行实验：stdout 只提取 perplexity 行追加到 log，stderr 直接输出到终端
        $HOME/.conda/envs/awq/bin/python model_quant.py \
            --model_name_or_path=${MODEL1} \
            --format=$FORMAT \
            --w_bits=4 \
            --a_bits=4 \
            --w_group_size=32 \
            --a_group_size=32 \
            --transform_class=hadamard \
            --w_observer=mse \
            --quantization_order=activation \
            --hadamard_group_size=64 \
            --dataset_name_or_path=c4 \
            --num_sequences=128 \
            --gptaq \
            --rel_damp=0.01 \
            --sequence_length=2048 \
            --dtype=bfloat16 \
            --fuse_global_scale \
            --amp \
            --eval_perplexity \
            --alpha=$alpha \
            --channel_sorting \
            | grep -E "^Wikitext-2 perplexity:|^C4 perplexity:" >> ${OUTPUT_DIR}/log.txt
        
        # 提取结果（sed 比 awk 更鲁棒，不受前面字段影响）
        WIKI_PPL=$(grep "^Wikitext-2 perplexity:" ${OUTPUT_DIR}/log.txt | sed 's/^Wikitext-2 perplexity: //')
        C4_PPL=$(grep "^C4 perplexity:" ${OUTPUT_DIR}/log.txt | sed 's/^C4 perplexity: //')
        
        if [ -n "$WIKI_PPL" ] && [ -n "$C4_PPL" ]; then
            STATUS="success"
        else
            WIKI_PPL="NA"
            C4_PPL="NA"
            STATUS="failed"
        fi
        
        # 记录到 CSV
        echo "${TIMESTAMP},${MODEL1},${FORMAT},${alpha},${penalty},${WIKI_PPL},${C4_PPL},${STATUS}" >> $RESULTS_FILE
        
        # 清理 GPU 缓存
        python -c "import torch; torch.cuda.empty_cache()"
        
        echo "Completed: alpha=$alpha, penalty=$penalty, Wiki PPL=$WIKI_PPL, C4 PPL=$C4_PPL"
        echo ""
    done
done

echo "========================================"
echo "All experiments completed!"
echo "Results saved to: $RESULTS_FILE"
echo "========================================"
