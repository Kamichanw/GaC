CUDA_VISIBLE_DEVICES=0 nohup python serial.py --path configs/qwen-3b.yaml --dataset_name humaneval > humaneval.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python serial.py --path configs/qwen-3b.yaml --dataset_name gsm8k --use_fewshot >> gsm8k.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python serial.py --path configs/qwen-3b.yaml --dataset_name cnndm > cnndm.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python serial.py --path configs/qwen-3b.yaml --dataset_name mmlu > mmlu.log 2>&1 &