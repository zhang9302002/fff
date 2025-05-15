"""
    File: cli_2process.py
    Description: This file demonstrates an implementation of a multiprocess Real-time Long Video System. With a multiprocess logging module.
        main process: CLI server I/O, LLM inference
        process-1: logger listener
        process-2: frame generator, 
        process-3: frame memory manager
    Author: Haoji Zhang, Haotian Liu (This code is based on https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py)
    Date Created: 2024-02-26
    Last Modified: 2024-03-01
"""
import argparse

import torch
import numpy as np
import time
import os

from torch.multiprocessing import Process, Queue, Manager
from transformers import TextStreamer, Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from decord import VideoReader
from datetime import datetime
from PIL import Image
from io import BytesIO

from qwen_vl_utils import process_vision_info

def main(args):

    model_path = args.model_path
    video_file = args.video_file
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda", 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)

    frame_paths = os.listdir(video_file)
    frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    frame_paths = [os.path.join(video_file, frame_path) for frame_path in frame_paths]
    inp = 'what is in the video?'
    inp = \
"""Please choose the correct answer from the options below, output the option letter (A, B, C, or D):
A. A person running a marathon and sharing their experience
B. A cooking tutorial showing how to make a special dish
C. A car review and test drive on a highway
D. A dog training session in a park"""
    messages = [
        {
            "role": "user",
            "content": [
                {   "type": "text", 
                    "text": inp,
                },
                {
                    "type": "video",
                    "video": frame_paths,
                    "max_frames": 64,
                }
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    text += "Best Option: ("
    inputs = processor(
        text=[text],
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=3,
            use_cache=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs = outputs[0].strip()
        print(f"{outputs}", end="\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="output/best_ckpt")
    parser.add_argument("--video-file", type=str, default="data_link/eval_video/videomme/frames/goyWFUzCqF4")

    # parser.add_argument("--log-file", type=str, default="server_cli.log")
    # parser.add_argument("--use_1process", action="store_true")
    # parser.add_argument("--video_fps", type=float, default=0.5)
    # parser.add_argument("--play_speed", type=float, default=1.0)
    # parser.add_argument("--flash_memory_dict", type=str, default=None)
    args = parser.parse_args()
    # args.flash_memory_dict = dict(
    #     flash_memory_temporal_length=120, 
    #     flash_memory_temporal_method='kmeans_ordered',
    #     flash_memory_temporal_poolsize=2,
    #     flash_memory_temporal_pca_dim=32,
    #     flash_memory_spatial_length=60,
    #     flash_memory_spatial_method='klarge_retrieve',
    # )
    args.model_path = 'ckpt/Qwen2-VL-7B-Instruct'
    main(args)
