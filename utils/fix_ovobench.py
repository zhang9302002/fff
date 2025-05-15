import os
import json

qa = json.load(open('/mnt/bn/fasterlmm/mlx/workspace/llama-vstream/data/eval_video/ovobench/test_qa.json'))
qa_dic = {x['id']: x for x in qa}

output = '/mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/ckpt_models/sft_lora_3200_qwen2vl_7b_maxframe240_reso4*224*224_max14000_kmeans_ordered120_sample60pos_bs64_lr8e-4/evaluation_max240_reso4*224*224/ovobench'
output2 = output.replace('ovobench', 'ovobench_2')
if not os.path.exists(output2):
    os.makedirs(output2)

for i in range(8):
    name = '8_' + str(i)
    data = []
    with open(os.path.join(output, name + '.json')) as f:
        for line in f:
            d = json.loads(line)
            if not 'answer' in d:
                d['answer'] = qa_dic[d['id']]['answer']
            if not 'a_type' in d:
                d['a_type'] = qa_dic[d['id']]['origin']['task']
            data.append(d)

    with open(os.path.join(output2, name + '.json'), 'w') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
