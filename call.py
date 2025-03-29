import importlib
from pprint import pprint
import numpy as np
import requests
import torch
from tqdm import tqdm

def warpped_sampling(prompts,  max_prompt_len):

    results = {
        "generated": [],
        "total_time": [],
        "num_tokens": [],
        "num_tokens_per_sec": [],
    }

    for prompt in tqdm(prompts, desc="Processing prompts"):
        url = "http://0.0.0.0:8000/api/generate/"

        data = {
            "messages_list": [
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            ],
            "max_new_tokens": max_prompt_len,
            "apply_chat_template": False,
        }

        response = requests.post(url, json=data).json()

        results["num_tokens"].append(response["num_tokens"])
        results["total_time"].append(response["total_time"])
        results["num_tokens_per_sec"].append(response["num_tokens_per_sec"])
        results["generated"].append(response["response"][0])

    return results

def init_dataset(dataset_name, dataset_size, use_fewshot):
    MyDataset = importlib.import_module(
        f"src.mydatasets.{dataset_name}.mydataset"
    ).MyDataset
    dataset = MyDataset(size=dataset_size, use_fewshot=use_fewshot)
    return dataset

def process_result(results, evaluate_func):

    results_stats = {
        "performance": evaluate_func(results["generated"]),
        "num_tokens_per_sec": np.mean(results["num_tokens_per_sec"]),
        "total_time": np.mean(results["total_time"]),
        "num_tokens": np.mean(results["num_tokens"]),
    }


    pprint(results_stats)
    
if __name__ == "__main__":

    torch.manual_seed(42)

    dataset = init_dataset("gsm8k", dataset_size=100, use_fewshot=True)
    prompts = dataset.get_prompts()

    results = warpped_sampling(prompts, max_prompt_len=256)

    process_result(results, dataset.evaluate)
