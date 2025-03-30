from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import importlib
import yaml
from pprint import pprint
import numpy as np
from tqdm import tqdm
import argparse
from transformers import DynamicCache


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def load_models(path):
    config = load_config(path)
    models = []
    tokenizers = []
    scores = []

    for model_cfg in config["CONFIG_API_SERVER"]:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_cfg["weight"], torch_dtype=torch.float16
            )
            .to("cuda:0")
            .eval()
        )

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["weight"])

        models.append(model)
        tokenizers.append(tokenizer)
        scores.append(model_cfg["score"])

    first_vocab = tokenizers[0].get_vocab()
    for tok in tokenizers[1:]:
        assert tok.get_vocab() == first_vocab, "All tokenizers must be identical"

    total_score = sum(scores)
    if config["NORM_TYPE_API_SERVER"] == "average":
        weights = [1 / len(models)] * len(models)
    else:
        weights = [s / total_score for s in scores]

    return models, tokenizers[0], weights


models, tokenizer, ensemble_weights = None, None, None


def ensemble_greedy_decode(input_text, max_new_tokens=20):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(models[0].device)
    current_ids = input_ids.clone()

    past_key_values = [None for _ in range(len(models))]
    cache_position = torch.arange(
        input_ids.shape[1], dtype=torch.int64, device="cuda:0"
    )
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            probs = []
            for i, model in enumerate(models):
                outputs = model(
                    current_ids,
                    use_cache=True,
                    past_key_values=past_key_values[i],
                    cache_position=cache_position,
                )
                probs.append(torch.softmax(outputs.logits[:, -1, :], -1))
                past_key_values[i] = outputs.past_key_values

        weighted_probs = torch.zeros_like(probs[0])
        for weight, prob in zip(ensemble_weights, probs):
            weighted_probs += weight * prob

        next_token = torch.argmax(weighted_probs, dim=-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_token], dim=-1)
        cache_position = cache_position[-1:] + 1

        if next_token.item() == tokenizer.eos_token_id:
            break

    end.record()
    torch.cuda.synchronize()

    total_time = start.elapsed_time(end) / 1000.0
    num_tokens = current_ids.shape[1] - input_ids.shape[1]
    return {
        "response": [tokenizer.decode(current_ids[0], skip_special_tokens=True)],
        "num_tokens": num_tokens,
        "total_time": total_time,
        "num_tokens_per_sec": num_tokens / total_time if total_time > 0 else 0,
    }


def warpped_sampling(prompts, max_prompt_len, port=None):
    results = {
        "generated": [],
        "total_time": [],
        "num_tokens": [],
        "num_tokens_per_sec": [],
    }

    for prompt in tqdm(prompts, desc="Processing prompts"):
        response = ensemble_greedy_decode(prompt, max_new_tokens=max_prompt_len)

        results["num_tokens"].append(response["num_tokens"])
        results["total_time"].append(response["total_time"])
        results["num_tokens_per_sec"].append(response["num_tokens_per_sec"])
        results["generated"].append(response["response"][0])

        current_speed = np.mean(results["num_tokens_per_sec"])
        print(f"Current speed: {current_speed:.2f} tokens/sec")

    return results


def init_dataset(dataset_name, dataset_size, use_fewshot):
    print(f"Initializing dataset: {dataset_name}")
    MyDataset = importlib.import_module(
        f"src.mydatasets.{dataset_name}.mydataset"
    ).MyDataset
    return MyDataset(size=dataset_size, use_fewshot=use_fewshot)


def process_result(results, evaluate_func):
    results_stats = {
        "performance": evaluate_func(results["generated"]),
        "num_tokens_per_sec": np.mean(results["num_tokens_per_sec"]),
        "total_time": np.mean(results["total_time"]),
        "num_tokens": np.mean(results["num_tokens"]),
    }
    pprint(results_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ensemble model decoding")
    parser.add_argument(
        "--dataset_name", type=str, default="gsm8k", help="Dataset name"
    )
    parser.add_argument("--dataset_size", type=int, default=5, help="Dataset size")
    parser.add_argument(
        "--use_fewshot", action="store_true", help="Use few-shot learning"
    )
    parser.add_argument("--path", type=str, help="Path to the YAML configuration file")

    args = parser.parse_args()
    torch.manual_seed(42)

    # Load models and tokenizer
    models, tokenizer, ensemble_weights = load_models(args.path)

    dataset = init_dataset(args.dataset_name, args.dataset_size, args.use_fewshot)
    results = warpped_sampling(dataset.get_prompts(), max_prompt_len=128)

    process_result(results, dataset.evaluate)
