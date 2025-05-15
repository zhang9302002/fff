
# !/bin/bash
set -x

source  /mnt/bn/fasterlmm/mlx/workspace/activate.sh
conda activate vstream
cd /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2

maxlen=14000
ngpus=8
pixel_expr=4*224*224
max_pixels=$((${pixel_expr}))
max_frames_train=240

t_length=120
t_method=kmeans_ordered
s_length=60
s_method=sample

suffix="maxframe${max_frames_train}_reso${pixel_expr}_max${maxlen}_${t_method}${t_length}_${s_method}${s_length}pos_bs64_lr8e-4"
suffix="reorganized"

DO_TRAIN=0
DO_EVAL=1

DISTRIBUTED_ARGS="
    --nproc_per_node ${ngpus} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6002
"
DATE="$(date +%m%d)"
OUTPUT_BASE=/mnt/bn/longvideockpt/flash_output_7b
OUTPUT_NAME=bestckpt_${suffix}
SAVE_PATH="${OUTPUT_BASE}/${OUTPUT_NAME}"
mkdir -p "$SAVE_PATH"


export OPENAIKEY=RN4PXs662q4VUamjRoubRP1NlU4YMH33
export OPENAIBASE=https://search.bytedance.net/gpt/openapi/online/v2/crawl
export OPENAITYPE=azure
export OPENAIVERSION=2023-03-15-preview
if [ $DO_EVAL -eq 1 ]; then
    # for max_frames in 40 60 120 240
    for max_frames in 240
    do
        suffix="max${max_frames}_reso${pixel_expr}_kmeans120klarge60"
        for dataset in egoschema lvbench
        do
            echo start eval ${dataset}
            python3 eval_any_dataset.py \
                --model-path ./output/best_ckpt \
                --dataset ${dataset} \
                --output_dir ${SAVE_PATH} \
                --evaluation_name evaluation_${suffix} \
                --num_chunks ${ngpus} \
                --max_frames ${max_frames} \
                --max_pixels ${max_pixels} \
                --flash_memory_dict '{"flash_memory_temporal_length":120,"flash_memory_temporal_method":"kmeans_ordered","flash_memory_temporal_poolsize":2,"flash_memory_spatial_length":60,"flash_memory_spatial_method":"klarge_retrieve"}' \
                --api_key ${OPENAIKEY} \
                --api_base ${OPENAIBASE} \
                --api_type ${OPENAITYPE} \
                --api_version ${OPENAIVERSION} \
                >> "${SAVE_PATH}/${DATE}_qwen2vl-7b-eval-${dataset}_${suffix}.log" 2>&1 
        done
    done
fi

bash /mnt/bn/fasterlmmlq/workspace/run.sh