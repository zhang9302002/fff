source  /mnt/bn/fasterlmm/mlx/workspace/activate.sh
conda activate vstream
cd /mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2

# python utils/extract_video_feature_1fps.py \
#     --video_dir data_link/eval_video/videomme/videos \
#     --gt_file data_link/eval_video/videomme/test_qa.json \
#     --output_file data_link/eval_video/videomme/scan_video_list.json \
#     --feature_dir data_link/eval_video/videomme/video_features_224_224 \
#     --num-chunks 1 \
#     --cmd scan


python utils/extract_video_feature_2fps.py \
    --video_dir ./data_link/llava-video-178k/frames \
    --gt_file ./data_link/llava-video-178k/frames \
    --output_file /mnt/bn/longvideo/dataset/LLaVA-Video-178K/videolist.json \
    --feature_dir /mnt/bn/longvideockpt/features/llava_video_178k_features_4_224_224 \
    --num-chunks 8 \
    --cmd extract \
    >> /mnt/bn/longvideockpt/features/llava_video_178k_features_4_224_224_2fps.log 2>&1

    


# python utils/extract_video_feature_2fps.py \
#     --video_dir ./data_link/llava-video-178k/frames \
#     --gt_file ./data_link/llava-video-178k/frames \
#     --output_file /mnt/bn/longvideo/dataset/LLaVA-Video-178K/err_list_feat.json \
#     --feature_dir /mnt/bn/longvideockpt/features/llava_video_178k_features_224_224_re2 \
#     --num-chunks 8 \
#     --cmd extract \
#     >> /mnt/bn/longvideockpt/features/videomme_extract_video_feature_2fps_re2.log 2>&1

