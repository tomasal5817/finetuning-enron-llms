# README

This repository is a part of the master thesis  **"Assessing Privacy vs. Efficiency Trade-offs in Open-Source Large Language Models"**, during 2025. It supports fine-tuning a range of open-source language models such as Qwen3-8B (https://huggingface.co/Qwen/Qwen3-8B) on the Enron email dataset (https://huggingface.co/datasets/LLM-PBE/enron-email/tree/main). //

Models fine-tuned during the master thesis using this code can be found on Hugging Face: https://huggingface.co/Tomasal 

## Setup environment:
To get started, create and activate a virtual environment with the required dependencies:
```bash
  - conda create -n finetune-env python=3.10
  - conda activate finetune-env
  - pip install -r requirements.txt
```
## Run Fine-tuning
python finetune.py \
  --dataset_path ./data/enron.json \
  --output_dir ./output \
  --model_name Qwen/Qwen3-8B \
  --use_lora True \
  --precision bf16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2

 
