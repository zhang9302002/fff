import decord
import torch
import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

min_frames = 4
max_frames = 768

def process_video(file):
    video_id = file.strip().split('.')[0]
    video_reader = decord.VideoReader(f'data/{file}')
    video_len = len(video_reader)
    duration = video_len / video_reader.get_avg_fps()
    nframes = round(duration) * 2
    nframes = min(max(nframes, min_frames), max_frames, video_len // 2 * 2)
    start_frame_ids = 0
    end_frame_ids = video_len - 1
    idx = torch.linspace(start_frame_ids, end_frame_ids, nframes).round().long().clamp(0, video_len - 1)
    frames = video_reader.get_batch(idx.tolist()).asnumpy()
    os.makedirs(f'video_{max_frames}frames_high/{video_id}', exist_ok=True)
    for i in range(len(frames)):
        img = Image.fromarray(frames[i])
        img.save(f'video_{max_frames}frames_high/{video_id}/frame_{i}.jpg', quality=100)


if __name__ == '__main__':
    files = [f for f in os.listdir('data')]
    with Pool(processes=96) as pool:
        list(tqdm(pool.imap_unordered(process_video, files), total=len(files)))

