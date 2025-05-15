# Based on https://github.com/haotian-liu/LLaVA.

import os
import json
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu

from flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from flash_vstream.conversation import conv_templates, SeparatorStyle
from flash_vstream.model.builder import load_pretrained_model
from flash_vstream.utils import disable_torch_init
from flash_vstream.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from qwen_vl_utils import process_vision_info


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


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

    parser.add_argument("--T_max_frames", type=int, default=180)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    return parser.parse_args()


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

    
def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        args.model_max_length,
        torch_dtype=torch.float16, 
        # torch_dtype=torch.bfloat16, 
        # attn_implementation="flash_attention_2",
        T_max_frames=args.T_max_frames
    )

    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            print(f'mkdir Except: {e}')

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

    for sample in tqdm(gt_questions, desc=f"cuda:{args.chunk_idx} "):
        QUESTION_PROMPT = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
        video_name = sample['video_id']
        question = QUESTION_PROMPT + sample['question']
        id = sample['id']
        answer = sample['answer']

        sample_set = {'id': id, 'question': question, 'answer': answer}

        video_path = os.path.join(args.video_dir, video_name)
        # Check if the video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist.")
    
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
                        "max_frames": 120,
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
        inputs = image_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            T_max_frames=args.T_max_frames,
        )
        print(f'video_inputs={len(video_inputs[0])} {np.array(video_inputs[0][0]).shape}, inputs_id={inputs.input_ids.shape}')
        inputs = inputs.to("cuda")

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                top_k=1,
                do_sample=False,
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

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
