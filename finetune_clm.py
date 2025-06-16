import os
import logging
import wandb
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    set_seed,
    GenerationConfig,
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
    resume_from_checkpoint: bool = field(default=False)
    model_name: str = field(default="Qwen/Qwen3-8B", metadata={"help": "Base model name"})
    dataset_path: str = field(default="", metadata={"help": "Path to JSON dataset with 'text'"})
    output_dir: str = field(default="./output", metadata={"help": "Directory to save the fine-tuned model"})
    use_lora: bool = field(default=True)
    precision: str = field(default="bf16", metadata={"help": "fp16 or bf16"})
    push_to_hub: bool = field(default=False)
    hub_token: Optional[str] = field(default=None)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)  # Reduced to prevent GPU memory overflow
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Batch size = per_device_train_batch_size *  gradient_accumulation_steps"}) 
    use_enron: bool = field(default=True)
    seed: int = field(default=42)
    block_size: int = field(default=512, metadata={"help": "Maximum sequence length per training example (in tokens)"})  

def load_enron_dataset():
    try:
        from enron import CustomEnron
    except ImportError:
        raise ImportError("The 'enron' module is required to load the Enron dataset. Please install it.")

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

    if args.use_enron:
        raw_datasets = load_enron_dataset()
    else:
        raw_datasets = load_dataset("json", data_files=args.dataset_path)
    
    if "train" not in raw_datasets:
        raise ValueError("Dataset must contain a 'train' split.")
    if "text" not in raw_datasets["train"].column_names:
        raise ValueError("The 'train' split must contain a 'text' field.")
    
    print_highlighted('Print the first example')
    print(raw_datasets["train"][0])  # Print the first example
    print_highlighted('Check column names')
    print(raw_datasets["train"].column_names)  # Check column names

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = (
        torch.bfloat16 if args.precision == "bf16"
        else torch.float16 if args.precision == "fp16"
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    #model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_lora:
        logger.info("Applying LoRA")
        print_highlighted("Applying LoRA")
        peft_config = LoraConfig(
            # Limit LoRA to attention layers
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"{name} is trainable but has no gradient!")
    else:
        logger.warning("Training full model without PEFT/LoRA.")

    model.train()
    model.enable_input_require_grads()
    print_highlighted(f"Model training mode: {model.training}")

    # Move the model to the GPU (optional with Trainer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Qwen3 8B tokenizer
    max_length = min(tokenizer.model_max_length, 2048)

    # Tokenz chat template 
    chat_temp_tokens = tokenizer(
    "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<|im_end|>",
    add_special_tokens=False
    )["input_ids"]
    reserved_length = len(chat_temp_tokens)

    def tokenize_function(examples):
        """
        Tokenizes input emails using the Qwen3 chat template.

        Formats each email as a user message in a chat but leaves the assistant's response block empty 
        Truncated to max 2048 tokens

        Note: The assistant’s answer is not included in the final tokenized output
        """
        formatted = []
        for email_text in examples["text"]:
            max_user_tokens = tokenizer.model_max_length - reserved_length
            tokenized_input = tokenizer(
                email_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_user_tokens
            )["input_ids"]

            truncated_email = tokenizer.decode(tokenized_input, skip_special_tokens=False)

            # Add Qwen3 chat template
            chat_format = (
                "<|im_start|>user\n"
                f"{truncated_email}<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<|im_end|>"
            )
            formatted.append(chat_format)

        # Tokenize the fully formatted chat strings
        return tokenizer(
            formatted,
            truncation=False,
            padding=False,
            max_length=max_length
        )

    tokenized_path = os.path.join(args.output_dir, f"tokenized_maxlength{max_length}")
    if os.path.exists(tokenized_path):
        print_highlighted("Loading tokenized dataset from disk")
        tokenized_datasets = load_from_disk(tokenized_path)
    else:
        print_highlighted("Tokenizing dataset...")
        tokenized_datasets = DatasetDict({
            split: raw_datasets[split].map(
                tokenize_function,
                batched=True,
                remove_columns=raw_datasets[split].column_names,
                num_proc=1
            )
            for split in raw_datasets
        })
        print_highlighted(f"Saving tokenized dataset to {tokenized_path}")
        tokenized_datasets.save_to_disk(tokenized_path)
    
    tokenizer.pad_token = tokenizer.eos_token

    for example in tokenized_datasets:
        if not example.strip():  
            print("Empty text:", example)

    def group_texts(examples, block_size):
        """
        Concatenates and chunks tokenized inputs into fixed-size blocks for training, reduces data-load 

        Note: Consider batching when training using a chat template as token chunking is more suited fore training causal LLMs 
        """
        concatenated_examples = {
            k: sum((ex for ex in examples[k] if ex), [])
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped_path = os.path.join(args.output_dir, f"lm_datasets_block{args.block_size}")

    if os.path.exists(grouped_path):
        print_highlighted(f"Loading grouped dataset from {grouped_path}")
        lm_datasets = load_from_disk(grouped_path)
    else:
        print_highlighted('Group text')
        lm_datasets = DatasetDict({
            split: tokenized_datasets[split].map(
                lambda examples: group_texts(examples, block_size=args.block_size),
                batched=True,
                num_proc=4
            )
            for split in tokenized_datasets
        })
        print_highlighted(f"Saving grouped dataset to {grouped_path}")
        lm_datasets.save_to_disk(grouped_path)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

   # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        run_name="fine-tuning-run",
        eval_strategy="steps",
        # evaluation_strategy="steps",
        eval_steps=50000,
        save_total_limit=2,
        save_strategy="steps",
        save_steps=50000,
        overwrite_output_dir=True,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        learning_rate=1e-4, # Typical Range: 1e-4 (0.0001) to 5e-5 (0.00005).
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.model_name if args.push_to_hub else None,
        report_to="wandb",
    )

    if args.push_to_hub:
        if not args.hub_token:
            raise ValueError("Hub token must be provided when pushing to the Hugging Face Hub.")
        login(token=args.hub_token)

    print(lm_datasets["train"][33])
    
    inputs = tokenizer("Test sentence", return_tensors="pt").to(model.device)
    inputs['labels'] = inputs['input_ids'].clone()
    outputs = model(**inputs)
    outputs.loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(name, param.grad.norm())
    
    print_highlighted("Starting fine-tuning...")

    eval_dataset = lm_datasets.get("validation", lm_datasets["test"])
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    print_highlighted(f"Model saved to {args.output_dir}")

    metrics = trainer.evaluate(eval_dataset=lm_datasets.get("test", eval_dataset))
    print(f"Test set evaluation: {metrics}")

    if args.push_to_hub:
        trainer.push_to_hub()
        print_highlighted("Model pushed to Hugging Face Hub")

    logger.info("Fine-tuning complete!")
    print_highlighted("Fine-tuning complete!")

if __name__ == "__main__":
    main()
