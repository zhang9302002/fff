import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import torch

with open('utils/test_speed.txt') as f:
    text_data = f.read()
# print(f'text_data={text_data}')
# print(f'text_length={len(text_data)}')

# default: Load the model on the available device(s)
device="cuda"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "ckpt/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)

# default processer
processor = AutoProcessor.from_pretrained("ckpt/Qwen2-VL-2B-Instruct")
print(f'model is loaded')
n_k = 14

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are a helpfule writing judger. Judge the writing quality of this article, A. Excellent B. Great C. Good D. Fair. \n" + text_data * n_k},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
text += 'Your option: ('
input_len = n_k * 1024
input_len = 10

output_len = 1
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    max_length=input_len,
    truncation=True,
)
inputs = inputs.to("cuda")
print(f'input length={inputs.input_ids.shape}')

# Inference: Generation of the output
times = 10
with torch.inference_mode():
    for _ in range(3):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=output_len,
        )
    start = time.perf_counter()
    for _ in range(times):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=output_len,
        )
    end = time.perf_counter()
    for _ in range(3):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=output_len,
        )
total_time = (end-start) / times
print(f'time={total_time}')

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
