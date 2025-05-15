# quick start

```
bash setup.sh  # create environment
bash run7b_iccv_kmeansord_sample60pos_20250306_bs64_lr8e-4.sh  # test on ovobench, you need to edit the model path, dataset config.

inference_mcq_vqa.py  # this is the inference code of Flash-VStream
```

# Customized Evaluation
By default, we use 120-frames-large memory, you can also change the eval code to use other memory size.
```
python3 eval_any_dataset.py \
    --model-path ${SAVE_PATH} \
    --dataset ${dataset} \
    --output_dir ${SAVE_PATH} \
    --evaluation_name evaluation_${suffix}_test0515 \
    --num_chunks ${ngpus} \
    --max_frames ${max_frames} \
    --max_pixels ${max_pixels} \
    --flash_memory_dict '{"flash_memory_temporal_length":120,"flash_memory_temporal_method":"kmeans_ordered","flash_memory_temporal_poolsize":2,"flash_memory_spatial_length":60,"flash_memory_spatial_method":"sample"}' \
    >> "${SAVE_PATH}/${DATE}_qwen2vl-7b-eval-${dataset}_${suffix}.log" 2>&1 
```

For example, here set `"flash_memory_temporal_length":240` will use 240-frames-large memory.