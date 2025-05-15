import csv
import openai
import os
import re
import ast
import json
import random
import argparse

from time import sleep
from multiprocessing.pool import Pool
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    parser.add_argument("--num_chunks", default=1, type=int, help="Result splits")
    parser.add_argument("--api_key", required=True, type=str, help="OpenAI API key")
    parser.add_argument("--api_type", default=None, type=str, help="OpenAI API type")
    parser.add_argument("--api_version", default=None, type=str, help="OpenAI API version")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI API base")
    args = parser.parse_args()
    return args


def extract_answer(llm_message):
    answer = re.findall(r'[A-E]', llm_message)
    if len(answer) == 0:
        print('No answer found')
        answer = random.choice(['A', 'B', 'C', 'D', 'E'])
    else:
        answer = answer[0]
    map2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    answer = map2idx[answer]
    return answer


def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    for file in tqdm(caption_files):
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        try:
            # Compute the correctness score
            res = extract_answer(pred)
            if res == answer:
                acc = "yes"
            else:
                acc = "no"
            response_dict = {"pred": acc, "score": 3}
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)
            # sleep(0.5)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")
            sleep(1)


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    if args.num_chunks > 1:
        pred_contents = []
        for _idx in range(args.num_chunks):
            file = os.path.join(args.pred_path, f"{args.num_chunks}_{_idx}.json")
            pred_contents += [json.loads(line) for line in open(file)]
        
    else:
        file = os.path.join(args.pred_path, f"pred.json")
        pred_contents = [json.loads(line) for line in open(file)]

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['id']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['id'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred, "a_type": sample['answer_type'] if 'answer_type' in sample else None}
        prediction_set[id] = qa_set

    # Set the OpenAI API key.
    openai.api_key = args.api_key # Your API key here
    if args.api_type:
        openai.api_type = args.api_type
    if args.api_version:
        openai.api_version = args.api_version
    if args.api_base:
        openai.api_base = args.api_base # Your API base here
    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    incomplete_lengths = []
    for _ in range(100):
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")
            incomplete_lengths.append(len(incomplete_files))
            if len(incomplete_lengths) > 5 and len(set(incomplete_lengths[-5:])) <= 1:
                print(f"incomplete_lengths: {incomplete_lengths}")
                print(f"incomplete_files: {incomplete_files}")
                print(f"completed_files: {completed_files}")
                print(f"failed for 5 times, break")
                break

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                assert 'pred' in content[0], f"Error: {file_name} don't has key=pred"
                assert 'score' in content[0], f"Error: {file_name} don't has key=score"
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    class ScoreMeter:
        def __init__(self):
            self.score_sum = 0
            self.count = 0
            self.yes_count = 0
            self.no_count = 0
            self.score_dict = {'yes': defaultdict(int), 'no': defaultdict(int)}

        def add_score(self, score, pred):
            self.score_sum += score
            self.count += 1
            pred_lower = pred.lower()
            if 'yes' in pred_lower:
                self.yes_count += 1
                self.score_dict['yes'][score] += 1
            elif 'no' in pred_lower:
                self.no_count += 1
                self.score_dict['no'][score] += 1

        def get_average_score(self):
            res = (self.score_sum / self.count) if self.count else 0
            return f"{res:.6f}"

        def get_accuracy(self, response_type):
            if response_type == 'yes':
                res =  (self.yes_count / self.count) if self.count else 0
            elif response_type == 'no':
                res = (self.no_count / self.count) if self.count else 0
            else:
                res = 0
            return f"{res:.6f}"

    meter_dic = {'total': ScoreMeter()}
    for key, result in combined_contents.items():
        # Computing score
        score_match = result[0]['score']
        score = int(score_match)
        pred = result[0]['pred']

        meter_dic["total"].add_score(score, pred)
        if 'a_type' in result[1] and result[1]['a_type'] is not None:
            typ = str(result[1]['a_type'])
            if typ not in meter_dic:
                meter_dic[typ] = ScoreMeter()
            meter_dic[typ].add_score(score, pred)

            if 'next' in args.output_dir:
                typ = typ[0]
                if typ not in meter_dic:
                    meter_dic[typ] = ScoreMeter()
                meter_dic[typ].add_score(score, pred)

    csv_dic = {'acc': meter_dic["total"].get_accuracy('yes'), 'score': meter_dic["total"].get_average_score()}

    # print("Yes count:", meter_dic["total"].yes_count)
    # print("No count:", meter_dic["total"].no_count)
    # print("Accuracy:", meter_dic["total"].get_accuracy('yes'))
    # print("Average score:", meter_dic["total"].get_average_score())
    # print("")
    # print("Total Score Yes/No distribution:")
    # for key, value in meter_dic["total"].score_dict.items():
    #     print(f"{key}:")
    #     for k in range(0, 6):
    #         v = value[k]
    #         print(f"{k}: {v}")
    # print("")
    # print("Answer Type Score distribution:")
    # print('Type, Accuracy, Avg_score')
    # key_list = sorted([k for k in meter_dic.keys()])
    # for key in key_list:
    #     print(key, meter_dic[key].get_accuracy('yes'), meter_dic[key].get_average_score(), sep=", ")
    #     csv_dic[key] = meter_dic[key].get_accuracy('yes')

    # print("")
    # for k in csv_dic.keys():
    #     print(k, end=", ")
    # print("")
    # for k in csv_dic.keys():
    #     print(csv_dic[k], end=", ")

    output = ""
    output += "Yes count: " + str(meter_dic["total"].yes_count) + "\n"
    output += "No count: " + str(meter_dic["total"].no_count) + "\n"
    output += "Accuracy: " + str(meter_dic["total"].get_accuracy('yes')) + "\n"
    output += "Average score: " + str(meter_dic["total"].get_average_score()) + "\n"
    output += "\n"
    output += "Total Score Yes/No distribution:\n"
    for key, value in meter_dic["total"].score_dict.items():
        output += f"{key}:\n"
        for k in range(0, 6):
            v = value[k]
            output += f"{k}: {v}\n"
    output += "\n"
    output += "Answer Type Score distribution:\n"
    output += 'Type, Accuracy, Avg_score\n'
    key_list = sorted([k for k in meter_dic.keys()])
    for key in key_list:
        output += f"{key}, {meter_dic[key].get_accuracy('yes')}, {meter_dic[key].get_average_score()}\n"
        csv_dic[key] = meter_dic[key].get_accuracy('yes')

    output += "\n"
    for k in csv_dic.keys():
        output += f"{k}, "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    for k in csv_dic.keys():
        output += str(csv_dic[k]) + ", "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    # kaggle upload
    if 'egoschema' in args.pred_path:
        with open(args.output_json.replace(".json", "_upload.csv"), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['q_uid', 'answer'])
            all_qa = json.load(open('/mnt/bn/fasterlmm/mlx/dataset/longvideo/egoschema/all_qa.json'))
            info_dic = {}
            for qa in all_qa:
                info_dic[str(qa['id'])] = qa['video_id']
            for key, result in combined_contents.items():
                pred = result[1]['pred']
                q_uid = info_dic[key.split('_')[0]]
                res = extract_answer(pred)
                writer.writerow([q_uid, res])
    elif 'videomme' in args.pred_path:
        score_dic = {
            "duration": {"short": ScoreMeter(), "medium": ScoreMeter(), "long": ScoreMeter()},
            "domain": {"Knowledge": ScoreMeter(), "Film & Television": ScoreMeter(), "Sports Competition": ScoreMeter(), "Artistic Performance": ScoreMeter(), "Life Record": ScoreMeter(), "Multilingual": ScoreMeter()}, 
            "sub_category": {"Humanity & History": ScoreMeter(), "Literature & Art": ScoreMeter(), "Biology & Medicine": ScoreMeter(), "Finance & Commerce": ScoreMeter(), "Astronomy": ScoreMeter(), "Geography": ScoreMeter(), "Law": ScoreMeter(), "Life Tip": ScoreMeter(), "Technology": ScoreMeter(), "Animation": ScoreMeter(), "Movie & TV Show": ScoreMeter(), "Documentary": ScoreMeter(), "News Report": ScoreMeter(), "Esports": ScoreMeter(), "Basketball": ScoreMeter(), "Football": ScoreMeter(), "Athletics": ScoreMeter(), "Other Sports": ScoreMeter(), "Stage Play": ScoreMeter(), "Magic Show": ScoreMeter(), "Variety Show": ScoreMeter(), "Acrobatics": ScoreMeter(), "Handicraft": ScoreMeter(), "Food": ScoreMeter(), "Fashion": ScoreMeter(), "Daily Life": ScoreMeter(), "Travel": ScoreMeter(), "Pet & Animal": ScoreMeter(), "Exercise": ScoreMeter(), "Multilingual": ScoreMeter()}, 
            "task_type": {"Temporal Perception": ScoreMeter(), "Spatial Perception": ScoreMeter(), "Attribute Perception": ScoreMeter(), "Action Recognition": ScoreMeter(), "Object Recognition": ScoreMeter(), "OCR Problems": ScoreMeter(), "Counting Problem": ScoreMeter(), "Temporal Reasoning": ScoreMeter(), "Spatial Reasoning": ScoreMeter(), "Action Reasoning": ScoreMeter(), "Object Reasoning": ScoreMeter(), "Information Synopsis": ScoreMeter()},
        }
        total_dic = ScoreMeter()
        test_qa = json.load(open('data/eval_video/videomme/test_qa.json'))
        info_dic = {}
        for qa in test_qa:
            info_dic[str(qa['id'])] = qa
        level_list = ['duration', 'domain', 'sub_category', 'task_type']
        for key, result in combined_contents.items():
            pred = result[0]['pred']
            qa = info_dic[key.split('_')[0]]
            for level in level_list:
                score_dic[level][qa[level]].add_score(0, pred)
            total_dic.add_score(0, pred)
        output += '\n'
        output += 'Type, Accuracy\n'
        for level in level_list:
            for key, meter_dic in score_dic[level].items():
                output += f"{key}, {float(meter_dic.get_accuracy('yes')) * 100:.02f}\n"
        output += f"Overall, {float(total_dic.get_accuracy('yes')) * 100:.02f}\n"

    print(output)
    args.output_csv = args.output_json.replace(".json", ".csv")
    with open(args.output_csv, 'w') as f:
        f.write(output)
        
if __name__ == "__main__":
    main()

