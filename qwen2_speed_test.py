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
import requests
import logging
from logging.handlers import QueueHandler

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

class _Metric:
    def __init__(self):
        self._latest_value = None
        self._sum = 0.0
        self._max = 0.0
        self._count = 0

    @property
    def val(self):
        return self._latest_value

    @property
    def max(self):
        return self._max

    @property
    def avg(self):
        if self._count == 0:
            return float('nan')
        return self._sum / self._count

    def add(self, value):
        self._latest_value = value
        self._sum += value
        self._count += 1
        if value > self._max:
            self._max = value

    def __str__(self):
        latest_formatted = f"{self.val:.6f}" if self.val is not None else "None"
        average_formatted = f"{self.avg:.6f}"
        max_formatted = f"{self.max:.6f}"
        return f"{latest_formatted} ({average_formatted}, {max_formatted})"
        

class MetricMeter:
    def __init__(self):
        self._metrics = {}

    def add(self, key, value):
        if key not in self._metrics:
            self._metrics[key] = _Metric()
        self._metrics[key].add(value)

    def val(self, key):
        metric = self._metrics.get(key)
        if metric is None or metric.val is None:
            raise ValueError(f"No values have been added for key '{key}'.")
        return metric.val

    def avg(self, key):
        metric = self._metrics.get(key)
        if metric is None:
            raise ValueError(f"No values have been added for key '{key}'.")
        return metric.avg

    def max(self, key):
        metric = self._metrics.get(key)
        if metric is None:
            raise ValueError(f"No values have been added for key '{key}'.")
        return metric.max

    def __getitem__(self, key):
        metric = self._metrics.get(key)
        if metric is None:
            raise KeyError(f"The key '{key}' does not exist.")
        return str(metric)


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
                    "max_frames": 10,
                    "max_pixels": 4 * 224 * 224,
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

    # 启动server
    start_time = datetime.now()
    time_meter = MetricMeter()
    conv_cnt = 0
    for _ in range(100):
        # 获取当前时间
        now = datetime.now()
        conv_start_time = time.perf_counter()
        # 将当前时间格式化为字符串
        current_time = now.strftime("%H:%M:%S")
        duration = now.timestamp() - start_time.timestamp()

        # 打印当前时间
        # print("\nCurrent Time:", current_time, "Run for:", duration)
        # print(f"user: {inp}", end="\n")
        # print(f"assistant: ", end="")
        # every conversation is a new conversation
        
        llm_start_time = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1,
                use_cache=False,
            )
        llm_end_time = time.perf_counter()
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs = outputs[0].strip()
        print(f"{outputs}", end="\n")
        conv_end_time = time.perf_counter()
        if conv_cnt > 0:
            time_meter.add('conv_latency', conv_end_time - conv_start_time)
            time_meter.add('llm_latency', llm_end_time - llm_start_time)
            time_meter.add('real_sleep', conv_start_time - last_conv_start_time)
            print(f'CliServer: idx={conv_cnt},\treal_sleep={time_meter["real_sleep"]},\tconv_latency={time_meter["conv_latency"]},\tllm_latency={time_meter["llm_latency"]}')
        else:
            print(f'CliServer: idx={conv_cnt},\tconv_latency={conv_end_time - conv_start_time},\tllm_latency={llm_end_time - llm_start_time}')
        conv_cnt += 1
        last_conv_start_time = conv_start_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="output/best_ckpt")
    parser.add_argument("--video-file", type=str, default="data_link/eval_video/videomme/frames/goyWFUzCqF4")

    parser.add_argument("--log-file", type=str, default="server_cli.log")
    parser.add_argument("--use_1process", action="store_true")
    parser.add_argument("--video_fps", type=float, default=0.5)
    parser.add_argument("--play_speed", type=float, default=1.0)

    parser.add_argument("--flash_memory_dict", type=str, default=None)
    args = parser.parse_args()
    args.flash_memory_dict = dict(
        flash_memory_temporal_length=120, 
        flash_memory_temporal_method='kmeans_ordered',
        flash_memory_temporal_poolsize=2,
        flash_memory_temporal_pca_dim=32,
        flash_memory_spatial_length=60,
        flash_memory_spatial_method='klarge_retrieve',
    )
    args.model_path = 'ckpt/Qwen2-VL-7B-Instruct'
    main(args)
