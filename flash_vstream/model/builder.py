#    This file may have been modified by Flash-VStream Authors (Flash-VStream Modifications”). All Flash-VStream Modifications are Copyright 2024 Flash-VStream Authors. 
# ------------------------------------------------------------------------
# Based on https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoProcessor
import torch
from flash_vstream.model import VStreamLlamaForCausalLM, VStreamConfig
from flash_vstream.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from flash_vstream.model.language_model.processing_qwen2vl import Qwen2VLProcessor

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    config = AutoConfig.from_pretrained(model_path)
    config.vision_config.update({'T_max_frames': kwargs.pop('T_max_frames')})

    model = VStreamLlamaForCausalLM.from_pretrained(
        model_path, config=config, low_cpu_mem_usage=True, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len
