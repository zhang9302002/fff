from lmms_eval.tasks.videomme.utils import VIDEO_TYPE, CATEGORIES, SUB_CATEGORIES, TASK_CATEGORIES
import json
import csv
import pandas as pd

def videomme_aggregate_results(results):
    """
    Args:
        results, a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_type = result["duration"]
        category = result["category"]
        sub_category = result["sub_category"]
        task_category = result["task_category"]
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"video Type, {video_type}, {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Categories, {category}, {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Video Sub Categories, {sub_cate}, {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Task Categories, {task_cate}, {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    print(f"Overall Performance, , {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")
    return 100 * total_correct / total_answered if total_answered > 0 else 0

def videomme_aggregate_4_results(results):
    """
    Args:
        results, a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for i, result in enumerate(results[0]):
        video_type = result["duration"]
        category = result["category"]
        sub_category = result["sub_category"]
        task_category = result["task_category"]
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        score = 0
        for j in range(len(results)):
            score += results[j][i]["pred_answer"] == results[j][i]["answer"]
        category2score[key]["correct"] += score >= 1

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"video Type, {video_type}, {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Categories, {category}, {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Video Sub Categories, {sub_cate}, {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Task Categories, {task_cate}, {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    print(f"Overall Performance, , {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%, {total_answered}")
    return 100 * total_correct / total_answered if total_answered > 0 else 0


path = '/mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/logs/0909_1437_qwen2...ame10_qwen2_vl_model_args_dd899d/videomme.json'
path = '/mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/logs/0909_1428_qwen2...ame20_qwen2_vl_model_args_581529/videomme.json'
path = '/mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/logs/0909_1430_qwen2...ame40_qwen2_vl_model_args_2cbdb0/videomme.json'
path = 'logs/0909_1433_qwen2...ame90_qwen2_vl_model_args_990ac6/videomme.json'

path = 'logs/videomme_ws/7b_frame10_180subtitle/0912_1156_qwen2_vl_qwen2_vl_model_args_dd899d/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame20_180subtitle/0912_1156_qwen2_vl_qwen2_vl_model_args_581529/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame40_180subtitle/0912_1156_qwen2_vl_qwen2_vl_model_args_2cbdb0/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame90_180subtitle/0912_1233_qwen2_vl_qwen2_vl_model_args_990ac6/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame180_180subtitle/0912_1246_qwen2_vl_qwen2_vl_model_args_107e16/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame360_180subtitle/0912_1314_qwen2_vl_qwen2_vl_model_args_cf9353/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame540_180subtitle/0913_0442_qwen2_vl_qwen2_vl_model_args_48edb5/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame720_180subtitle/0913_0536_qwen2_vl_qwen2_vl_model_args_2cadb0/videomme_w_subtitle.json'

path = 'logs/videomme_ws/7b_frame10_180subtitle_randorder/0925_1005_qwen2_vl_qwen2_vl_model_args_dd899d/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame20_180subtitle_randorder/0925_1005_qwen2_vl_qwen2_vl_model_args_581529/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame40_180subtitle_randorder/0925_1018_qwen2_vl_qwen2_vl_model_args_2cbdb0/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame90_180subtitle_randorder/0925_1018_qwen2_vl_qwen2_vl_model_args_990ac6/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame180_180subtitle_randorder/0925_1116_qwen2_vl_qwen2_vl_model_args_944bb3/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame360_180subtitle_randorder/0925_1137_qwen2_vl_qwen2_vl_model_args_606da5/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame540_180subtitle_randorder/0925_1211_qwen2_vl_qwen2_vl_model_args_ed1919/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame720_180subtitle_randorder/0925_1258_qwen2_vl_qwen2_vl_model_args_a309a4/videomme_w_subtitle.json'

path = 'logs/videomme_ws/7b_frame720_180subtitle_shift1/0925_2046_qwen2_vl_qwen2_vl_model_args_a309a4/videomme_w_subtitle.json'

path = 'logs/videomme_ws/7b_frame900_180subtitle/0927_0859_qwen2_vl_qwen2_vl_model_args_7aad70/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame1200_180subtitle/0927_0913_qwen2_vl_qwen2_vl_model_args_d7c9e4/videomme_w_subtitle.json'
path = 'logs/videomme_ws/7b_frame1800_180subtitle/0927_0915_qwen2_vl_qwen2_vl_model_args_1aa9a5/videomme_w_subtitle.json'

d = json.load(open(path))
result = [x['videomme_percetion_score'] for x in d['logs']]

long_path = 'models/Qwen2-VL-7B-Instruct-vstream/evaluation/videomme_frame3600_long/pred.json'
with open(long_path) as f:
    text = f.readlines()
n = 900
rows = []
for i in range(n):
    rows.append(json.loads('\n'.join([text[i * 6 + j] for j in range(6)])))
long_dic = {}
for row in rows:
    long_dic[row['id']] = row['pred'][0]
for i in range(len(result)):
    if result[i]['question_id'] in long_dic:
        result[i]['pred_answer'] = long_dic[result[i]['question_id']]
        assert result[i]['duration'] == 'long'

# paths = [
#     'logs/videomme_ws/7b_frame10_180subtitle/0912_1156_qwen2_vl_qwen2_vl_model_args_dd899d/videomme_w_subtitle.json',
#     'logs/videomme_ws/7b_frame10_180subtitle/0918_0036_qwen2_vl_qwen2_vl_model_args_dd899d/videomme_w_subtitle.json',
#     'logs/videomme_ws/7b_frame10_180subtitle/0918_0045_qwen2_vl_qwen2_vl_model_args_dd899d/videomme_w_subtitle.json',
#     'logs/videomme_ws/7b_frame10_180subtitle/0918_0052_qwen2_vl_qwen2_vl_model_args_dd899d/videomme_w_subtitle.json',
#     # 'logs/videomme_ws/7b_frame10_180subtitle/0918_0100_qwen2_vl_qwen2_vl_model_args_dd899d/videomme_w_subtitle.json',
# ]
# results = []
# for path in paths:
#     d=json.load(open(path))
#     result = [x['videomme_percetion_score'] for x in d['logs']]
#     results.append(result)

# result = [x for x in result if x['duration'] == 'long']

videomme_aggregate_results(result)
# videomme_aggregate_4_results(results)

# PHASE = 'short'
# res = []
# rows = []
# rows.append(['duration', 'task_type', 'domain', 'sub_domain', 'url', 'question', 'answer', 'pred_answer'])
# for x in d['logs']:
#     if x['videomme_percetion_score']['duration'] == PHASE and x['videomme_percetion_score']['pred_answer'] != x['videomme_percetion_score']['answer']:
#         y = x['videomme_percetion_score']
#         y['doc'] = x['doc']
#         y['doc']['pred_answer'] = x['videomme_percetion_score']['pred_answer']
#         res.append(y)
#         q = y['doc']['question'] + "\n" + '\n'.join(y['doc']['options'])
#         rows.append([y['doc']['duration'], y['doc']['task_type'], y['doc']['domain'], y['doc']['domain'], y['doc']['url'], q, y['doc']['answer'], y['doc']['pred_answer']])

# with open(f'utils/720_ws_{PHASE}_wrong.json', 'w') as f:
#     json.dump(res, f, indent=4)

# with open(f'utils/720_ws_{PHASE}_wrong.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(rows)

# df = pd.DataFrame(rows[1:], columns=rows[0])
# df.to_excel(f'utils/720_ws_{PHASE}_wrong.xlsx', index=False)

# res = []
# for x in d['logs']:
#     if x['videomme_percetion_score']['sub_category'] == "Basketball" and x['videomme_percetion_score']['pred_answer'] != x['videomme_percetion_score']['answer']:
#         y = x['videomme_percetion_score']
#         y['doc'] = x['doc']
#         y['doc']['pred_answer'] = x['videomme_percetion_score']['pred_answer']
#         res.append(y)

# with open(f'utils/basketball_wrong.json', 'w') as f:
#     json.dump(res, f, indent=4)

# # Temporal Reasoning
# res = []
# for x in d['logs']:
#     if x['videomme_percetion_score']['task_category'] == "Temporal Reasoning" and x['videomme_percetion_score']['pred_answer'] != x['videomme_percetion_score']['answer']:
#         y = x['videomme_percetion_score']
#         y['doc'] = x['doc']
#         y['doc']['pred_answer'] = x['videomme_percetion_score']['pred_answer']
#         res.append(y)

# with open(f'utils/temporal_reasoning_wrong.json', 'w') as f:
#     json.dump(res, f, indent=4)

# # Counting Problem
# res = []
# for x in d['logs']:
#     if x['videomme_percetion_score']['task_category'] == "Counting Problem" and x['videomme_percetion_score']['pred_answer'] != x['videomme_percetion_score']['answer']:
#         y = x['videomme_percetion_score']
#         y['doc'] = x['doc']
#         y['doc']['pred_answer'] = x['videomme_percetion_score']['pred_answer']
#         res.append(y)

# with open(f'utils/counting_problem_wrong.json', 'w') as f:
#     json.dump(res, f, indent=4)