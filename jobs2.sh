
set -x

# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run7b_19200mcq_lora_qwen2vl_2b_frm120_1x224x224_onlypredans_bs64_lr2e-6.sh
# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run2b_19200mcq_lora_qwen2vl_2b_frm120_1x224x224_onlypredans_bs64_lr2e-6.sh


# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/old_cmds_1104/run2b_3200_lora_qwen2vl_2b_frm120_1x224x224_onlypredans_bs32_lr1e-4_e3.sh
# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/old_cmds_1104/run2b_3200_lora_qwen2vl_2b_frm120_1x224x224_onlypredans_bs32_lr5e-4_e3.sh
# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/old_cmds_1104/run2b_3200_lora_qwen2vl_2b_frm120_1x224x224_onlypredans_bs256_lr1e-4_e3.sh
# bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/old_cmds_1104/run2b_3200_lora_qwen2vl_2b_frm120_1x224x224_onlypredans_bs256_lr1e-3_e3.sh
bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run7b_merge.sh
bash /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/run7b_drop.sh
bash /mnt/bn/fasterlmmlq/workspace/run.sh
