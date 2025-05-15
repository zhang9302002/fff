
# !/bin/bash
set -x

source  /mnt/bn/fasterlmm/mlx/workspace/activate.sh
conda activate vstream
cd /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2

maxlen=18600
ngpus=8
pixel_expr=1*224*224
max_pixels=$((${pixel_expr}))

max_frames_train=600
suffix="maxframe${max_frames_train}_reso${pixel_expr}"

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
OUTPUT_NAME=Qwen2-VL-7B-Instruct-vstream
SAVE_PATH="${OUTPUT_BASE}/${OUTPUT_NAME}"
mkdir -p "$SAVE_PATH"

export NCCL_DEBUG=WARN
if [ $DO_TRAIN -eq 1 ]; then
    echo "start finetune, write to ${SAVE_PATH}"
    torchrun $DISTRIBUTED_ARGS finetune.py \
        --model_name_or_path ./ckpt/Qwen2-VL-7B-Instruct \
        --data_path ./data_train/videomme.json \
        --video_path ./videomme/video_frames_fps1_ori \
        --bf16 True \
        --fix_vit False \
        --output_dir ${SAVE_PATH} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 30 \
        --save_total_limit 30 \
        --learning_rate 5e-5 \
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

    suffix="_reproduce_64frm"
    for dataset in ovobench
    # for dataset in egoschema_all
    # for dataset in lvbench egoschema_all
    # for dataset in videommewo videommesub
    do
        echo start eval ${dataset}
        python3 eval_any_dataset.py \
            --model-path ${SAVE_PATH} \
            --dataset ${dataset} \
            --output_dir ${SAVE_PATH} \
            --evaluation_name evaluation_${suffix} \
            --num_chunks ${ngpus} \
            --reproduce \
            --reproduce_total_pixels $((10800*28*28)) \
            >> "${SAVE_PATH}/${DATE}_qwen2vl-7b-eval-${dataset}_${suffix}.log" 2>&1 
    done
fi

# bash /mnt/bn/fasterlmmlq/workspace/run.sh
