import os
import json
import copy
import argparse

from time import sleep
from tqdm import tqdm
from functools import reduce
from subprocess import Popen, PIPE

import torch
import multiprocessing
from threading import Thread


from torch.utils.data import Dataset
from decord import VideoReader
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def cut_video(arg):
    rootpath, outpath, name = arg
    video_path = os.path.join(rootpath, name)
    fps = 2
    vid = os.path.join(outpath, name.split(".")[0])
    if os.path.exists(video_path):
        if not os.path.exists(vid):
            os.makedirs(vid)
        os.system(f'ffmpeg -i {video_path} -q 0 -r {fps} {vid}/%06d.jpg')  # -r 5代表每秒抽取5帧。删除该参数即默认全部帧
    else:
        print("File not exists {}".format(video_path))
        return

def run_extract_frames(args):
    print(f'start run_extract')
    gt_questions = json.load(open(args.gt_file))
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    print(f'gt_questions loaded')
    all_videos = []
    for sample in tqdm(gt_questions, desc=f"cuda:{args.chunk_idx} "):
        if 'video_id' in sample:
            video_name = sample['video_id']
        else:
            video_name = 'v_' + sample['video_name']  # ActivityNet format

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            tmp_name = f"{video_name}{fmt}"
            if os.path.exists(temp_path):
                video_path = temp_path
                video_name = tmp_name
                break
        # Check if the video exists
        if os.path.exists(video_path):
            all_videos.append(video_name)
    
    all_videos_sorted = sorted(set(all_videos))
    print(f'video folder list ready., len={len(all_videos)}, lensorted={len(all_videos_sorted)}')
    # cut_video((args.video_dir, args.frame_dir, all_videos_sorted[0]))
    arg_list = []
    for name in tqdm(all_videos_sorted):
        arg_list.append((args.video_dir, args.frame_dir, name))

    with torch.multiprocessing.Pool() as pool:
        pool.map(cut_video, arg_list)

def get_video_set(param):
    question_list, worker_id = param
    video_set = set()
    video_formats = ['.mp4']
    not_found_set = set()
    for sample in tqdm(question_list, desc=f"worker:{worker_id}"):
        if 'video' in sample:
            video_path = os.path.join(args.video_dir, sample['video'])
            if os.path.exists(video_path):
                video_set.add(sample['video'])
            else:
                not_found_set.add(sample['video'])
        elif 'video_id' in sample:
            finded = False
            for fmt in video_formats:  # Added this line
                video_path = os.path.join(args.video_dir, sample['video_id'] + fmt)
                if os.path.exists(video_path):
                    video_set.add(sample['video_id'] + fmt)
                    finded = True
                    break
            if not finded:
                not_found_set.add(sample['video_id'])

    return video_set, not_found_set

def add(res_a, res_b):
    return res_a[0].union(res_b[0]), res_a[1].union(res_b[1])

def run_scan_video_set(args):
    print(f'start scan')
    gt_questions = json.load(open(args.gt_file))

    print(f'gt_questions loaded')
    video_questions = []
    for sample in tqdm(gt_questions, desc=f"main"):
        if 'video' in sample:
            video_questions.append(sample)
        elif 'video_id' in sample:
            video_questions.append(sample)
    
    print(f'video_questions loaded, total {len(video_questions)}')
    chunk_len = len(video_questions) // args.num_chunks

    param_list = []
    for i in range(0, len(video_questions), chunk_len):
        end = min(i + chunk_len, len(video_questions))
        chunk_list = video_questions[i:end]
        param_list.append((chunk_list, i // chunk_len))

    with torch.multiprocessing.Pool() as pool:
        res_list = pool.map(get_video_set, param_list)
    video_set, not_found_set = reduce(add, res_list)

    print(f'scan video folder {args.gt_file} finished, len={len(video_set)}, notfound={len(not_found_set)}')

    video_list = list(video_set)
    video_list.sort()
    not_found_list = list(not_found_set)
    not_found_list.sort()
    with open(args.output_file, 'w') as f:
        json.dump({'video_list': video_list, 'not_found_list': not_found_list}, f)

def run_convert_video_type(args):

    video_list = json.load(open(args.output_file))['not_found_list']
    video_formats = ['.avi', '.mov', '.mkv', '.webm']
    plist = []
    for video_name in tqdm(video_list):
        real_path = None
        for fmt in video_formats:  # Added this line
            # temp_path = os.path.join(args.video_dir, f"{video_name[:-4]}{fmt}")
            temp_path = os.path.join(args.video_dir, f"{video_name[:]}{fmt}")
            if os.path.exists(temp_path):
                real_path = temp_path
                break
        if real_path is not None:
            out_path = real_path[:-4] + '.mp4'
            if os.path.exists(out_path):
                continue
            ffmpeg_cmd = ['ffmpeg', '-i', real_path, out_path]
            process = Popen(ffmpeg_cmd)
            plist.append(process)
            sleep(1)
    for p in plist:
        p.wait()

class VideoDataset(Dataset):
    def __init__(self, video_dir, video_list, processor):
        self.video_dir = video_dir
        self.video_list = video_list
        
        self.processor = processor
        self.fps = 2.0
        self.max_frames = 3600
        self.max_pixels = 4 * 224 * 224

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        video_path = os.path.join(self.video_dir, video_name[:-4])
        
        frame_paths = os.listdir(video_path)
        frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        frame_paths = [os.path.join(video_path, frame_path) for frame_path in frame_paths]
        print(f'Dataloader, video_len={len(frame_paths)}')
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_paths,
                        "fps": self.fps,
                        "max_frames": self.max_frames,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": "query"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "answer"},
                ],
            }
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        video_inputs = self.processor.image_processor(images=None, videos=video_inputs, return_tensors="pt")

        return video_inputs["pixel_values_videos"], video_inputs["video_grid_thw"][0], video_name
        

def run_extract_feature(args, video_list):
    device = torch.device(f"cuda:{args.chunk_idx}")
    torch.cuda.set_device(device)

    model_path = 'ckpt/Qwen2-VL-7B-Instruct'
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{args.chunk_idx}",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    dataset = VideoDataset(args.video_dir, video_list, processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f'[cuda:{args.chunk_idx}]: dataloader loaded, len={len(dataloader)}')
    thread_list = []
    for batch in tqdm(dataloader, desc=f"cuda:{args.chunk_idx} "):
        pixel_values_videos, video_grid_thw, video_name = batch
        pixel_values_videos = pixel_values_videos.to(device)
        video_grid_thw = video_grid_thw.to(device)
        video_name = video_name[0]

        out_path = os.path.join(args.feature_dir, video_name.replace('.mp4', '.safetensors'))

        farther_path = os.path.dirname(out_path)
        if not os.path.exists(farther_path):
            os.makedirs(farther_path)
        with torch.no_grad():
            pixel_values_videos = pixel_values_videos.type(model.visual.get_dtype())
            image_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)

        save_file(
            {
                'image_embeds': image_embeds.cpu().detach().contiguous(),
                'video_grid_thw': video_grid_thw.cpu().detach().contiguous()
            }, 
            out_path
        )

        # save_thread = Thread(target=save_file, args=({'image_embeds': image_embeds}, out_path))
        # save_thread.start()
        print(f'Saving video, length={len(image_embeds)}, shape={image_embeds.shape}, name={video_name}. ')

        # thread_list.append(save_thread)
    
        # if len(thread_list) >= args.num_threads:
        #     print(f'Clearing threads...')
        #     for thread in thread_list:
        #         thread.join()
        #     thread_list = []
            
    for thread in thread_list:
        thread.join()

def main_extract(args):
    video_list = json.load(open(args.output_file))

    unprocessed_list = []
    for video_name in tqdm(video_list):
        if not os.path.exists(os.path.join(args.feature_dir, video_name.replace('.mp4', '.safetensors'))):
            unprocessed_list.append(video_name)
    video_list = unprocessed_list
    print(f'video_list loaded, len={len(video_list)}')

    if args.num_chunks == 1:
        args.chunk_idx = 0
        args.mm_vision_select_layer = -2
        run_extract_feature(args, video_list)
        return

    chunk_len = len(video_list) // args.num_chunks
    processes = []

    for i in range(args.num_chunks):
        arg = copy.copy(args)
        arg.chunk_idx = i
        arg.mm_vision_select_layer = -2
        video_li = video_list[arg.chunk_idx * chunk_len:(arg.chunk_idx + 1) * chunk_len]
        if i == args.num_chunks - 1:
            video_li = video_list[arg.chunk_idx * chunk_len:]

        process = torch.multiprocessing.Process(target=run_extract_feature, args=(arg, video_li))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print(f"{len(processes)} Processes have finished execution.")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Path containing video files.', required=True)
    parser.add_argument('--frame_dir', help='Path to write frame files.', required=False)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=False)
    parser.add_argument('--output_file', help='Path to video list file.', required=False)
    parser.add_argument('--feature_dir', help='Path to write video features.', required=False)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=10)
    parser.add_argument("--cmd", type=str, choices=['scan', 'extract', 'convert'])

    args = parser.parse_args()

    # run_extract_frames(args)

    if args.cmd == 'scan':
        run_scan_video_set(args)
    elif args.cmd == 'convert':
        run_convert_video_type(args)
    elif args.cmd == 'extract':
        torch.multiprocessing.set_start_method('spawn')
        main_extract(args)
    
