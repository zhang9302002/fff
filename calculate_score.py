import os
import torch
import json
# import jsonl
import transformers
import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, pipeline
import argparse
import re
import ast
import math
import json_lines
import argparse
import requests
import pandas as pd

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    # parser.add_argument('--s', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--data_set", type=str, required=False, default='msvd')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)

    return parser.parse_args()

def main(args):
    output_dir = args.output_dir
    output_name = args.output_name
    
    res_dict = []
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    
    files = os.listdir(output_dir)
    for file in files:
        if file.startswith(output_name):
            continue
        with open(f"{output_dir}/{file}", "rb") as f:
            # combined_contents = json_lines.reader(f)
            for result in tqdm.tqdm(json_lines.reader(f)):
                # Calculate average score and accuracy
                # for result in tqdm.tqdm(combined_contents):
                try:
                    # Computing score
                    count += 1
                    score_match = result['score']
                    score = int(score_match)
                    score_sum += score

                    # Computing accuracy
                    pred = result['llama_pred']
                    if "yes" in pred.lower():
                        yes_count += 1
                    elif "no" in pred.lower():
                        no_count += 1
                except:
                    print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)
    
    res_dict.append({"Yes count":yes_count})
    res_dict.append({"No count":no_count})
    res_dict.append({"Accuracy":accuracy})
    res_dict.append({"Average score":average_score})
    
    with open(f"{output_dir}/{output_name}_res.json","w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)