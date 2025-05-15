import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from torchsummaryX import summary
import torch

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

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

video_path = "./data/eval_video/videomme/video_frames/goyWFUzCqF4"
frame_paths = os.listdir(video_path)
frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
frame_paths = [os.path.join(video_path, frame_path) for frame_path in frame_paths]
messages = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            # },
            {
                "type": "video",
                "video": frame_paths,
                "fps": 1,
                "max_frames": 120,
                "max_pixels": 8 * 224 * 224,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(f'text={text}')
text += 'Best option: ('
print(f'text2={text}')
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# summary(model, torch.ones([1, 100], dtype=torch.long, device=device))
ids = inputs.pop('input_ids')
summary(model, ids, **inputs)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
