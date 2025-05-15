import csv
import json
from collections import defaultdict

def count_inversions(arr):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count

frame_list = [
    10, 20, 40, 90, 180
]
file_list = [
    'models/Qwen2-VL-7B-Instruct-vstream/eval224/evaluation_frame_10/egoschema/results.json',
    'models/Qwen2-VL-7B-Instruct-vstream/eval224/evaluation_frame_20/egoschema/results.json',
    'models/Qwen2-VL-7B-Instruct-vstream/eval224/evaluation_frame_40/egoschema/results.json',
    'models/Qwen2-VL-7B-Instruct-vstream/eval224/evaluation_frame_90/egoschema/results.json',
    'models/Qwen2-VL-7B-Instruct-vstream/eval224/evaluation_frame_90/egoschema_180_224/results.json'
]

rows = defaultdict(list)

for frames, file in zip(frame_list, file_list):
    d = json.load(open(file))
    for k, v in d.items():
        v[0]['frames'] = frames
        rows[k].append(v)


rows_2 = []
for k, v in rows.items():
    rows_2.append([
        k,
        *[1 if x[0]['pred'] == 'yes' else 0 for x in v]
    ])

csv_file = "look.csv"

# 写入CSV文件
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['qid', *frame_list, 'inversions'])

#     for row in rows_2:
#         num = count_inversions(row[1:])
#         writer.writerow(row + [num])

# print(f"Data successfully written to {csv_file}")

look_list = [
    '13_0', '1574_0', '1887_0'
]
d = {k:rows[k] for k in look_list}
json_file = "look.json"

# 写入JSON文件
with open(json_file, mode='w', encoding='utf-8') as file:
    json.dump(d, file, indent=4, ensure_ascii=False)