# CUDA_VISIBLE_DEVICES=2 nohup python serial.py --path configs/llama-vicuna.yaml --dataset_name humaneval > humaneval.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python serial.py --path configs/llama-vicuna.yaml --dataset_name gsm8k --use_fewshot > gsm8k.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python serial.py --path configs/llama-vicuna.yaml --dataset_name cnndm > cnndm.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python serial.py --path configs/llama-vicuna.yaml --dataset_name mmlu > mmlu.log 2>&1 &