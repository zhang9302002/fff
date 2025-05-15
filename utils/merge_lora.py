import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
from models.vstream_qwen2vl import FlashVStreamQwen2VLModel, FlashVStreamQwen2VLConfig, get_real_grid_thw, get_spatial_real_grid_thw
from peft import AutoPeftModelForCausalLM
import ast
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from qwen_vl_utils import process_vision_info


parser = argparse.ArgumentParser()
parser.add_argument('--qwen_path', type=str, required=True)
parser.add_argument('--lora_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

qwen_path = args.qwen_path
tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

lora_path = args.lora_path
model = AutoPeftModelForCausalLM.from_pretrained(lora_path, device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16).eval()

print("Load Success")
model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(qwen_path)

model = model.merge_and_unload()
print(f'model type = {type(model)}')

output_path = args.output_path
os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
processor.save_pretrained(output_path)
print(f"Model and tokenizer saved to {output_path}")

chat_template_path = os.path.join(output_path, "chat_template.json")
with open(chat_template_path, 'w') as f:
    json.dump({'chat_template': processor.chat_template}, f, indent=4)
print(f"Chat template saved to {chat_template_path}")
