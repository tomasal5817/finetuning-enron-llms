import os
import logging
import wandb
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    set_seed,
)

from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from accelerate import Accelerator

def print_highlighted(text):
    print("\033[92m" + text + "\033[0m")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FineTuneArgs:
    model_name: str = field(default="Qwen/Qwen2.5-14B-Instruct", metadata={"help": "Base model name"})
    dataset_path: str = field(default="", metadata={"help": "Path to JSON dataset with 'text'"})
    output_dir: str = field(default="./output", metadata={"help": "Where to save model"})
    use_lora: bool = field(default=True)
    precision: str = field(default="bf16", metadata={"help": "fp16 nanor bf16"})
    push_to_hub: bool = field(default=False)
    hub_token: Optional[str] = field(default=None)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)
    use_enron: bool = field(default=True)
    seed: int = field(default=42)
    block_size: int = field(default=-1, metadata={"help": "Block size for sequences"})

def load_enron_dataset():
    from enron import CustomEnron
    logger.info("Loading Enron dataset...")
    print_highlighted("Loading Enron dataset...")
    enron_builder = CustomEnron()
    enron_builder.download_and_prepare()
    ds = enron_builder.as_dataset()
    logger.info("Enron dataset loaded!")
    print_highlighted("Enron dataset loaded!")
    return ds

def main():
    parser = HfArgumentParser(FineTuneArgs)
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator()
    logger.info(f"Using device: {accelerator.device}")
    print_highlighted(f"Using device: {accelerator.device}")
    logger.info(f"Setting seed to: {args.seed}")
    set_seed(args.seed)

    # Load dataset
    if args.use_enron:
        raw_datasets = load_enron_dataset()
        #train_dataset = raw_datasets["train"]
    else:
        raw_datasets = load_dataset("json", data_files=args.dataset_path)

    if "train" not in raw_datasets or "text" not in raw_datasets["train"].column_names:
        raise ValueError("Dataset must contain a 'text' field in the 'train' split.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-set block size
    if args.block_size == -1:
        args.block_size = min(tokenizer.model_max_length, 2048)
    
        logger.info(f"Auto-detected block size: {args.block_size}")
        print_highlighted(f"Auto-detected block size: {args.block_size}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16
    )

    # Optional: enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Apply LoRA if needed
    if args.use_lora:
        logger.info("Applying LoRA")
        print_highlighted("Applying LoRA")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
    else:
        logger.warning("Training full model without PEFT/LoRA. This may require substantial GPU memory.")

    # Tokenize and chunk
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length= 2048     #tokenizer.model_max_length
        )

    print_highlighted('Tokenize Dataset')

    tokenized_datasets = {
        split: raw_datasets[split].map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets[split].column_names,
            num_proc=4
        )
        for split in raw_datasets
    }

    def group_texts(examples, block_size):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop remainder to make perfect chunks
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        # Chunk into blocks of `block_size`
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        # Copy input_ids to labels for causal LM
        result["labels"] = result["input_ids"].copy()
        return result

    # Group texts
    print_highlighted('Group text')
    lm_datasets = {
        split: tokenized_datasets[split].map(
            lambda examples, _: group_texts(examples, block_size=args.block_size),
            group_texts,
            batched=True,
            num_proc=4
        )
        for split in tokenized_datasets
    }
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=2,
        save_strategy="steps",
        save_steps=5000,
        overwrite_output_dir=True,
        logging_strategy="steps",      
        logging_steps=10,
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.model_name if args.push_to_hub else None,
        report_to="wandb",
    )

    if args.push_to_hub and args.hub_token:
        login(token=args.hub_token)

    print_highlighted("Starting fine-tuning...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],  
        tokenizer=tokenizer,
        data_collator=data_collator,
        #callbacks=[GradientNormCallback(log_every=100)]
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    print_highlighted(f"Model saved to {args.output_dir}")

    metrics = trainer.evaluate(eval_dataset=lm_datasets["test"])
    print_highlighted(f"Test set evaluation: {metrics}")

    if args.push_to_hub:
        trainer.push_to_hub()
        print_highlighted("Model pushed to Hugging Face Hub")

    logger.info("Fine-tuning complete!")
    print_highlighted("Fine-tuning complete!")

if __name__ == "__main__":
    main()
