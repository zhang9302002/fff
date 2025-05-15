import json
import os
from tqdm import tqdm

info_path = '/mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/data_link/eval_video/videomme/test.json'
info = json.load(open(info_path))
info_dic = {x['question_id']: x for x in info}

paths = {
    'qwen2-vl': '/mnt/bn/longvideockpt/flash_output_7b/Qwen2-VL-7B-Instruct/evaluation_max180_reso2*224*224/videommewo',
    'llava-ov': '/mnt/bn/fasterlmmlq/workspace/LLaVA-NeXT/ckpt/LLaVA-Video-7B-Qwen2/evaluation/videomme_wo_frames60',
    'flash-vstream': '/mnt/bn/longvideockpt/flash_output_7b/sft_lora_3200_qwen2vl_7b_maxframe240_reso4*224*224_max14000_kmeans_ordered120_sample60pos_bs64_lr8e-4/evaluation_max240_reso4*224*224_klarge30_real_tight/videommewo',
}

datas = []
for name, path in paths.items():
    dataset = 'videomme'
    print(f'>>> model={name}, dataset={dataset}')
    datas.append(json.load(open(path + f'/result.json')))

corr_list = []
corr_a, corr_b = [], []
wrong_list = []
video_base = '/mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/data_link/eval_video/videomme/frames'
for key in tqdm(datas[0].keys()):
    a = datas[0][key]
    b = datas[1][key + '_0']
    c = datas[2][key]
    b_score = 1 if b[0]['pred'] == 'yes' else 0
    if a['score'] < c['score'] and b_score < c['score']:
    # if b_score < c['score']:
        corr_list.append([a, b])
        assert a['id'] == c['id']
        id = a['id']
        qa = info_dic[id].copy()
        qa.update({'pred_a': a['pred']})
        qa.update({'pred_b': b[1]['pred']})
        qa.update({'pred_c': c['pred']})
        qa.update({'video_path': os.path.join(video_base, qa['videoID'])})
        corr_b.append(qa)
    if a['score'] > c['score'] and b_score > c['score']:
        wrong_list.append([a, b])
        assert a['id'] == c['id']
        id = a['id']
        qa = info_dic[id].copy()
        qa.update({'pred_a': a['pred']})
        qa.update({'pred_b': b[1]['pred']})
        qa.update({'pred_c': c['pred']})
        qa.update({'video_path': os.path.join(video_base, qa['videoID'])})
        corr_a.append(qa)
    
print(f'corr len = {len(corr_list)}')
print(f'wrong len = {len(wrong_list)}')

out_path = f'./utils/case_study_{dataset}'
with open(out_path + f'_corr_b.json', 'w') as f:
    json.dump(corr_b, f, indent=4)

with open(out_path + f'_wrong_a.json', 'w') as f:
    json.dump(corr_a, f, indent=4)