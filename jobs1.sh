
set -x

# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run7b_19200mcq_lora_qwen2vl_2b_frm120_1x224x224_onlypredans_bs64_lr1e-5.sh
# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run2b_19200mcq_lora_qwen2vl_2b_frm120_1x224x224_onlypredans_bs64_lr1e-5.sh

# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run7b_evalqwen2.sh
# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run7b_baseline.sh
# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run7b_kmeans.sh

bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run_qwen2vl_7b_eval_frm360_224x224.sh
bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run_qwen2vl_7b_eval_frm180_2x224x224.sh
bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run_qwen2vl_7b_eval_frm90_4x224x224.sh
bash /mnt/bn/fasterlmmlq/workspace/run.sh