import ast
import json
from matplotlib import pyplot as plt
import numpy as np
import openai

# openai.api_key = 'RN4PXs662q4VUamjRoubRP1NlU4YMH33'
# openai.api_type = 'azure'
# openai.api_version = '2024-01-25-preview'
# openai.api_base = 'https://search.bytedance.net/gpt/openapi/online/v2/crawl'
# question = ''
# answer = ''
# pred = ''
# completion = openai.ChatCompletion.create(
#     engine="gpt-3.5-turbo-0125",
#     messages=[
#         {
#             "role": "system",
#             "content": 
#                 "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
#                 "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
#                 "------"
#                 "##INSTRUCTIONS: "
#                 "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
#                 "- Consider synonyms or paraphrases as valid matches.\n"
#                 "- Evaluate the correctness of the prediction compared to the answer."
#         },
#         {
#             "role": "user",
#             "content":
#                 "Please evaluate the following video-based question-answer pair:\n\n"
#                 f"Question: {question}\n"
#                 f"Correct Answer: {answer}\n"
#                 f"Predicted Answer: {pred}\n\n"
#                 "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
#                 "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
#                 "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
#                 "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
#         }
#     ],
#     temperature=0.002
# )
# # Convert response to a Python dictionary.
# print(f'completion={completion}')
# response_message = completion["choices"][0]["message"]["content"]
# response_dict = ast.literal_eval(response_message)

import torch
from tqdm import tqdm
from models.compress_functions import (
    drop_feature, merge_feature, kmeans_feature, weighted_kmeans_feature, weighted_kmeans_ordered_feature, 
    k_drop_feature, k_merge_feature, dbscan_feature, gmm_feature
)
import json
import torch.nn.functional as F


path = '/mnt/bn/longvideockpt/flash_output_7b/sft_lora_3200_qwen2vl_7b_maxframe240_reso1*224*224_max5000_kmeans120/test.json'
with open(path, 'r') as f:
    data = json.load(f)
# print(data['shape'])
x = data['feature']
x = torch.tensor(x, device="cuda")  # [91, 216, 1280]
print(f'x={x.shape}')

x = x.reshape(-1, 216*1280)
# min_vals = x.min()
# max_vals = x.max()
# x = (x - min_vals) / (max_vals - min_vals)
x = F.normalize(x, p=2, dim=1)

dist = ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(dim=2).sqrt() #/ torch.sqrt(torch.tensor(216))
print(f'dist={dist.shape} {dist}')

x = x.reshape(-1, 216, 1280)
# x, _, _ = dbscan_feature(x, 60)
x, _, _ = gmm_feature(x, 60)
print(f'x.shape={x.shape}')

