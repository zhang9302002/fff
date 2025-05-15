source  /mnt/bn/fasterlmm/mlx/workspace/activate.sh
conda activate vstream
cd /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2

suffix=model_7b_test30
# OUTPUT_BASE=/mnt/bn/longvideockpt/flash_output_7b/speed_test_2gpu
OUTPUT_BASE=.
mkdir -p $OUTPUT_BASE
python cli_server_2gpu.py \
    --model-path output/best_ckpt \
    --log-file ${OUTPUT_BASE}/server_cli_${suffix}.log 

