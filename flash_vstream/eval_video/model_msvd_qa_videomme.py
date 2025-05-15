# Based on https://github.com/haotian-liu/LLaVA.

import os
import json
import math
import re
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
# from vllm import LLM, SamplingParams


from flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from flash_vstream.conversation import conv_templates, SeparatorStyle
from flash_vstream.model.builder import load_pretrained_model
from flash_vstream.utils import disable_torch_init
from flash_vstream.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from qwen_vl_utils import process_vision_info

from accelerate import infer_auto_device_map


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    # chunk_size = math.ceil(len(lst) / n)  # integer division
    # return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    res = [[] for i in range(n)]
    for i, x in enumerate(lst):
        res[i % n].append(x)
    return res


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--model-max-length", type=int, default=None)

    parser.add_argument("--max_frames", type=int, default=180)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--use_subtitle", action="store_true", default=False)
    return parser.parse_args()


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame

    
def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    # model = LLM(
    #     args.model_path,
    #     dtype=torch.float16, 
    #     tensor_parallel_size=4,
    #     # max_model_len=131072,
    #     # pipeline_parallel_size=2,
    # )
    # device_map = {
    #     'visual.patch_embed': 0, 'visual.rotary_pos_emb': 0, 'visual.merger': 3, 
    #     'visual.blocks.0': 0, 'visual.blocks.1': 0, 'visual.blocks.2': 0, 'visual.blocks.3': 0,
    #     'visual.blocks.4': 0, 'visual.blocks.5': 0, 'visual.blocks.6': 0, 'visual.blocks.7': 0,
    #     'visual.blocks.8': 1, 'visual.blocks.9': 1, 'visual.blocks.10': 1, 'visual.blocks.11': 1,
    #     'visual.blocks.12': 1, 'visual.blocks.13': 1, 'visual.blocks.14': 1, 'visual.blocks.15': 1,
    #     'visual.blocks.16': 2, 'visual.blocks.17': 2, 'visual.blocks.18': 2, 'visual.blocks.19': 2,
    #     'visual.blocks.20': 2, 'visual.blocks.21': 2, 'visual.blocks.22': 2, 'visual.blocks.23': 2,
    #     'visual.blocks.24': 3, 'visual.blocks.25': 3, 'visual.blocks.26': 3, 'visual.blocks.27': 3,
    #     'visual.blocks.28': 3, 'visual.blocks.29': 3, 'visual.blocks.30': 3, 'visual.blocks.31': 3,
    #     'model.embed_tokens': 4, 'model.layers.0': 4, 'model.layers.1': 4, 'model.layers.2': 4, 
    #     'model.layers.3': 4, 'model.layers.4': 4, 'model.layers.5': 4, 'model.layers.6': 4, 'model.layers.7': 4, 
    #     'model.layers.8': 5, 'model.layers.9': 5, 'model.layers.10': 5, 'model.layers.11': 5, 'model.layers.12': 5, 
    #     'model.layers.13': 5, 'model.layers.14': 5, 'model.layers.15': 5, 'model.layers.16': 5, 'model.layers.17': 5, 
    #     'model.layers.18': 6, 'model.layers.19': 6, 'model.layers.20': 6, 'model.layers.21': 6, 'model.layers.22': 6, 
    #     'model.layers.23': 6, 'model.layers.24': 6, 'model.layers.25': 6, 'model.layers.26': 6, 'model.layers.27': 6, 'model.norm': 6, 
    #     'lm_head': 7,
    # }

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        args.model_max_length,
        # torch_dtype=torch.float16, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
    )

    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)
    # gt_questions = [x for x in gt_questions if x['duration'] == 'long']  ##### only long video is tested

    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            print(f'mkdir Except: {e}')\

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            id_set = [json.loads(row)['id'] for row in f.readlines()]
            id_set = set(id_set)
            gt_questions = [sample for sample in gt_questions if sample['id'] not in id_set]

    ans_file = open(answers_file, "a")
    input_list = []
    sample_list = []
    
    for sample in tqdm(gt_questions, desc=f"cuda:{args.chunk_idx} preparing"):
        video_name = sample['video_id']
        SUBTITLE_PROMPT = "This video's subtitles are listed below: \n"
        QUESTION_PROMPT = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
        data_path = os.path.join("/mnt/bn/fasterlmmlq/dataset/Video-MME/data", video_name + ".mp4")
        subtitle_path = os.path.join("/mnt/bn/fasterlmmlq/dataset/Video-MME/subtitle", video_name + ".srt")
        
        if os.path.exists(subtitle_path):  # Denote have subtitle
            print(f'try to open{subtitle_path}')
            subtitle = open(subtitle_path, encoding='utf-8').readlines()
        else:
            subtitle = ""
        if subtitle == "":
            subtitle = "No subtitles available"
        else:
            frame_num = 180
            subtitle_by_frame, total_frame = extract_subtitles(data_path, subtitle_path)
            uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

            subtitle_by_frame_idx = []
            for frame_idx in uniform_sampled_frames:
                for idx, title in enumerate(subtitle_by_frame):
                    if frame_idx < title[1] and frame_idx >= title[0]:
                        subtitle_by_frame_idx.append(idx)
            subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

            textlist = []
            for idx in subtitle_by_frame_idx:
                pattern = r'<font color="white" size=".72c">(.*?)</font>'
                raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                try:
                    textlist.append(raw_text[0])
                except:
                    continue
            subtitle = "\n".join(textlist)

        ##### Use VideoMME-w/o subtitle
        if args.use_subtitle:
            question = SUBTITLE_PROMPT + subtitle + '\n' + QUESTION_PROMPT + sample['question']
        else:
            question = QUESTION_PROMPT + sample['question']
        id = sample['id']
        answer = sample['answer']
        video_path = os.path.join(args.video_dir, video_name)
        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        frame_paths = os.listdir(video_path)
        frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        frame_paths = [os.path.join(video_path, frame_path) for frame_path in frame_paths]
    
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_paths,
                        "fps": 1,
                        "max_frames": args.max_frames,
                        "max_pixels": args.max_pixels,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = image_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text += 'Best option: ('
        image_inputs, video_inputs = process_vision_info(messages)
        
        # mm_data = {}
        # if image_inputs is not None:
        #     mm_data["image"] = image_inputs
        # if video_inputs is not None:
        #     mm_data["video"] = video_inputs
        # llm_inputs = {
        #     "prompt": text,
        #     "multi_modal_data": mm_data,
        # }
        # sampling_params = SamplingParams(
        #     temperature=0.1,
        #     top_p=0.001,
        #     repetition_penalty=1.05,
        #     max_tokens=256,
        #     stop_token_ids=[],
        # )
        # with torch.inference_mode():
        #     outputs = model.generate([llm_inputs], sampling_params=sampling_params)
        # outputs = outputs[0].outputs[0].text.strip()

        inputs = image_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # print(f'video_inputs={len(video_inputs[0])} {np.array(video_inputs[0][0]).shape}, inputs_id={inputs.input_ids.shape}')
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                use_cache=True,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = image_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs = outputs[0].strip()

        print(f'outputs={outputs}')
        sample_set['pred'] = outputs
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

        # except Exception as e:
        #     print(f'Except: {e}')
        #     print(f'video_name: {video_name}')
        #     continue
    ans_file.close()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
