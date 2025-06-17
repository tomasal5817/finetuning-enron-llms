# README

This repository is part of the master's thesis  **"Assessing Privacy vs. Efficiency Trade-offs in Open-Source Large Language Models"**, during 2025. It supports fine-tuning a range of open-source language models such as [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on the [Enron email dataset](https://huggingface.co/datasets/LLM-PBE/enron-email/tree/main).

Models fine-tuned during the master's thesis can be found on Hugging Face Hub: https://huggingface.co/Tomasal 

## Setup environment:
To get started, create and activate a virtual environment with the required dependencies:
```bash
conda create -n finetune-env python=3.10
conda activate finetune-env
pip install -r requirements.txt
```
## Command-Line Arguments

Below are the configurable arguments for `finetune_clm.py`:

| Argument                        | Type    | Default           | Description                                                                 |
|---------------------------------|---------|-------------------|-----------------------------------------------------------------------------|
| `--resume_from_checkpoint`      | `bool`  | `False`           | Resume training from last checkpoint                                        |
| `--model_name`                  | `str`   | `Qwen/Qwen3-8B`   | Base model name (from Hugging Face)                                         |
| `--dataset_path`                | `str`   | `""`              | Path to a dataset in JSON format (ignored if using Enron)                   |
| `--output_dir`                  | `str`   | `./output`        | Output directory for model and logs                                         |
| `--use_lora`                    | `bool`  | `True`            | Use LoRA for parameter-efficient tuning                                     |
| `--precision`                   | `str`   | `"bf16"`          | One of: `fp16`, `bf16`, `fp32`                                              |
| `--push_to_hub`                 | `bool`  | `False`           | If true, pushes model to Hugging Face Hub                                   |
| `--hub_token`                   | `str`   | `None`            | Hugging Face token (required if pushing to hub)                             |
| `--num_train_epochs`            | `int`   | `3`               | Number of training epochs                                                   |
| `--per_device_train_batch_size` | `int`   | `2`               | Training batch size per device                                              |
| `--per_device_eval_batch_size`  | `int`   | `2`               | Evaluation batch size per device                                            |
| `--gradient_accumulation_steps` | `int`   | `1`               | Simulated larger batch size via gradient accumulation                       |
| `--use_enron`                   | `bool`  | `True`            | If true, loads and tokenizes the Enron dataset                              |
| `--seed`                        | `int`   | `42`              | Sets the seed to make results repeatable                                    |
| `--block_size`                  | `int`   | `512`             | Chunk length for language modeling token blocks                             |



## Run Fine-tuning
To start fine-tuning, run the following command:
```bash
python finetune_clm.py \
--dataset_path ./data/enron.json \
--output_dir ./output \
--model_name Qwen/Qwen3-8B \
--use_lora True \
--precision bf16 \
--num_train_epochs 3 \
--per_device_train_batch_size 2
```

 
