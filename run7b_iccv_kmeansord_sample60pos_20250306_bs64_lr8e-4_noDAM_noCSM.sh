
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
# OUTPUT_BASE=/mnt/bn/longvideockpt/flash_output_7b
OUTPUT_BASE=/mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/ckpt_models
OUTPUT_NAME=sft_lora_3200_qwen2vl_7b_${suffix}
SAVE_PATH="${OUTPUT_BASE}/${OUTPUT_NAME}"
mkdir -p "$SAVE_PATH"

export NCCL_DEBUG=WARN
if [ $DO_TRAIN -eq 1 ]; then
    echo "start finetune, write to ${SAVE_PATH}"
    torchrun $DISTRIBUTED_ARGS finetune_flash.py \
        --model_name_or_path ./ckpt/Qwen2-VL-7B-Instruct \
        --data_path ./data_link/llava-video-178k/trainset_exist_4_3200len.json \
        --video_path ./data_link/llava-video-178k/frames \
        --use_flash_attn True \
        --flash_memory_temporal_length ${t_length} \
        --flash_memory_temporal_method ${t_method} \
        --flash_memory_temporal_poolsize 2 \
        --flash_memory_spatial_length ${s_length} \
        --flash_memory_spatial_method ${s_method} \
        --bf16 True \
        --output_dir ${SAVE_PATH} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 10 \
        --save_total_limit 30 \
        --learning_rate 8e-4 \
        --weight_decay 0.1 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.01 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --model_max_length ${maxlen} \
        --max_frames ${max_frames_train} \
        --max_pixels ${max_pixels} \
        --lazy_preprocess True \
        --use_lora \
        --lora_r 64 \
        --lora_alpha 32 \
        --gradient_checkpointing \
        --deepspeed deepspeed/zero2_config.json \
        >> "${SAVE_PATH}/lora_qwen2vl_7b_${suffix}.log" 2>&1
fi

if [ $DO_EVAL -eq 1 ]; then
    # for max_frames in 40 60 120 240
        
    max_frames_train=120
    ngpus=1
    for max_frames in ${max_frames_train}
    do
        suffix="max${max_frames}_reso${pixel_expr}_noDAM_noCSM"
        # for dataset in videommewo
        # for dataset in egoschema lvbench mvbench videommewo
        # for dataset in ovobench

        for dataset in streambench
        do
            echo start eval ${dataset}
            python3 eval_any_dataset.py \
                --model-path ${SAVE_PATH} \
                --dataset ${dataset} \
                --output_dir ${SAVE_PATH} \
                --evaluation_name evaluation_${suffix} \
                --num_chunks ${ngpus} \
                --max_frames ${max_frames} \
                --max_pixels ${max_pixels} \
                --flash_memory_dict '{"flash_memory_temporal_length":120,"flash_memory_temporal_method":"kmeans_ordered","flash_memory_temporal_poolsize":2,"flash_memory_spatial_length":0,"flash_memory_spatial_method":"sample"}' \
                # --test \
                # --testtest \
                >> "${SAVE_PATH}/${DATE}_qwen2vl-7b-eval-${dataset}_${suffix}.log" 2>&1 
        done

        
    done
fi

# bash /mnt/bn/fasterlmmlq/workspace/run.sh