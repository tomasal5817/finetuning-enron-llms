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
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct", metadata={"help": "Base model name"})
    dataset_path: str = field(default="", metadata={"help": "Path to JSON dataset with 'text'"})
    output_dir: str = field(default="./output", metadata={"help": "Where to save model"})
    use_lora: bool = field(default=True)
    precision: str = field(default="fp16", metadata={"help": "fp16 or bf16"})
    push_to_hub: bool = field(default=False)
    hub_token: Optional[str] = field(default=None)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=32)  # Reduced to prevent GPU memory overflow
    per_device_eval_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Batch size = per_device_train_batch_size *  gradient_accumulation_steps"}) #TODO:
    use_enron: bool = field(default=True)
    seed: int = field(default=42)
    block_size: int = field(default=2048, metadata={"help": "Block size for sequences"})

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

    model.gradient_checkpointing_enable()
    model.config.use_cache = False  
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_lora:
        logger.info("Applying LoRA")
        print_highlighted("Applying LoRA")
        peft_config = LoraConfig(
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

    max_length = min(tokenizer.model_max_length, 2048)
    '''
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            add_special_tokens=False,  # No BOS/EOS from tokenizer
            padding=False,
            truncation=False,         # Handle truncation manually
            max_length=max_length,
        )

        for i, line in enumerate(tokenized["input_ids"]):

            #line = [tokenizer.bos_token_id] + line 

            # Truncate manuallye
            if len(line) > max_length-1:
                line = line[:max_length-1]
            
            line.append(tokenizer.eos_token_id)
            tokenized["input_ids"][i] = line
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized
    '''
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            add_special_tokens=False, 
            padding=False,
            truncation=True,
            max_length=max_length - 1  # Space for EOS
        )

        for i, line in enumerate(tokenized["input_ids"]):
            line.append(tokenizer.eos_token_id)
            tokenized["input_ids"][i] = line

        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_path = os.path.join(args.output_dir, f"tokenized_maxlength{max_length}")
    if os.path.exists(tokenized_path):
        print_highlighted("Loading tokenized dataset from disk")
        lm_datasets = load_from_disk(tokenized_path)
    else:
        print_highlighted("Tokenizing dataset...")
        lm_datasets = DatasetDict({
            split: raw_datasets[split].map(
                tokenize_function,
                batched=True,
                remove_columns=raw_datasets[split].column_names,
                num_proc=1
            )
            for split in raw_datasets
        })

        print_highlighted(f"Saving tokenized dataset to {tokenized_path}")
        lm_datasets.save_to_disk(tokenized_path)
    print('DO WE GET HERE?')
    print(lm_datasets["train"][0])

    #lengths = [len(example["input_ids"]) for example in lm_datasets["train"]]
    #print(f"Min: {min(lengths)}, Max: {max(lengths)}")

    '''
    for example in lm_datasets:
        if not example.strip():  
            print("Empty text:", example)
    '''
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

   # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        run_name="fine-tuning-run",
        eval_strategy="steps",
        # evaluation_strategy="steps",
        eval_steps=5000,
        save_total_limit=2,
        save_strategy="steps",
        save_steps=5000,
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

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    print_highlighted(f"Model saved to {args.output_dir}")

    metrics = trainer.evaluate(eval_dataset=lm_datasets.get("test", eval_dataset))
    print_highlighted(f"Test set evaluation: {metrics}")

    if args.push_to_hub:
        trainer.push_to_hub()
        print_highlighted("Model pushed to Hugging Face Hub")

    logger.info("Fine-tuning complete!")
    print_highlighted("Fine-tuning complete!")

if __name__ == "__main__":
    main()
