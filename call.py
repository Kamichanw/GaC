import importlib
from pprint import pprint
import numpy as np
import requests
import torch
from tqdm import tqdm
import argparse

def warpped_sampling(prompts, max_prompt_len, port):
    results = {
        "generated": [],
        "total_time": [],
        "num_tokens": [],
        "num_tokens_per_sec": [],
    }

    for prompt in tqdm(prompts, desc="Processing prompts"):
        url = f"http://0.0.0.0:{port}/api/generate/"

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

        current_speed = response["num_tokens_per_sec"]
        print(f"Current speed: {current_speed:.2f} tokens/sec")

    return results

def init_dataset(dataset_name, dataset_size, use_fewshot):
    print(f"Initializing dataset: {dataset_name}")
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
    parser = argparse.ArgumentParser(description="Run prompt sampling and processing")
    parser.add_argument('--port', type=int, default=8000, help='Port number for the API')
    parser.add_argument('--dataset_name', type=str, default="gsm8k", help='Name of the dataset')
    parser.add_argument('--dataset_size', type=int, default=5, help='Size of the dataset')
    parser.add_argument('--use_fewshot', action='store_true', help='Use few-shot learning (default: False)')

    args = parser.parse_args()

    torch.manual_seed(42)

    dataset = init_dataset(args.dataset_name, dataset_size=args.dataset_size, use_fewshot=args.use_fewshot)
    prompts = dataset.get_prompts()

    results = warpped_sampling(prompts, max_prompt_len=256, port=args.port)

    process_result(results, dataset.evaluate)
