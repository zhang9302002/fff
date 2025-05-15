import os
import torch
import json
import transformers
import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, pipeline
import argparse
import re
import ast
import math

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--predict_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--llama3_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)

    return parser.parse_args()


def prepare_prompt(tokenizer:AutoTokenizer, messages:list):
    # tokens = []
    complete_message = []
    complete_message.append("<|begin_of_text|>")
    for messgae in messages:
        complete_message.append("<|start_header_id|>")
        complete_message.append(messgae["role"])
        complete_message.append("<|end_header_id|>")
        complete_message.append("\n\n")
        complete_message.append(messgae["content"])
        complete_message.append("<|eot_id|>")
    
    complete_message = " ".join(complete_message)
    message_ids = tokenizer(complete_message)
    return complete_message, message_ids

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def main(args):
    llama_path = args.llama3_path
    predict_file = args.predict_file
    output_dir = args.output_dir
    output_name = args.output_name
    kwargs = {"device_map": "auto"}
   
    kwargs['torch_dtype'] = torch.float16
    
    
    print("loading llama3 model for eval ...")
    llama_config = LlamaConfig.from_pretrained(llama_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
    llama_model = LlamaForCausalLM.from_pretrained(llama_path, low_cpu_mem_usage=True, config=llama_config, **kwargs)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    print(llama_model.device)
    print("llama3 load finish  !!")
    
    with open(os.path.join(predict_file, output_name + '.json')) as f:
        new_pred_contents = [json.loads(line) for line in f]

    os.makedirs(output_dir, exist_ok=True)
    answer_file = open(f"{output_dir}/{output_name}.json", "w")
    
    combined_contents = []
    # count = 0
    for pred in tqdm.tqdm(new_pred_contents, desc="Eval Video with LLaMA-3"):
        # count = count + 1
        # print(pred)
        question   = pred["question"]
        answer     = pred["answer"]
        prediction = pred["pred"]
        
        messages=[
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the correctness of the prediction compared to the answer."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer}\n"
                    f"Predicted Answer: {prediction}\n\n"
                    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                    "Please generate the response in the form of a Python dictionary string with keys 'llama_pred' and 'score', where value of 'llama_pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'llama_pred': 'yes', 'score': 4.8}."
            }
        ]
        complete_message, message_ids = prepare_prompt(llama_tokenizer, messages)
        ids = torch.tensor(message_ids['input_ids'], dtype=torch.long).to(llama_model.device)
        attention_mask = torch.tensor(message_ids['attention_mask']).to(llama_model.device).unsqueeze(0)
        embeddings = llama_model.model.embed_tokens(ids).unsqueeze(0).to(dtype=torch.float16)
        
        with torch.inference_mode():
            output_ids = llama_model.generate(
                inputs_embeds = embeddings,
                attention_mask = attention_mask,
                pad_token_id=llama_tokenizer.eos_token_id,
            )
            
        out_text = llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        matches = re.findall(r'\{.*?\}', out_text)
        for match in matches:
            result_dict = ast.literal_eval(match)
        pred.update(result_dict)
        combined_contents.append(pred)
        answer_file.write(json.dumps(pred) + "\n")

    print("Prediction complete")    
        
if __name__ == "__main__":
    args = parse_args()
    main(args)