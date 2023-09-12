import os
import argparse
import re
import json
import random
from datetime import datetime

import torch
import datasets

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

import wandb


def get_config():
    p = argparse.ArgumentParser()

    p.add_argument("--input_fn", type=str, required=True)
    p.add_argument("--output_dir_path", type=str, default="./checkpoints")
    p.add_argument("--pretrained_model_name", type=str, default="EleutherAI/polyglot-ko-12.8b")
    p.add_argument("--model_name", type=str, required=True)

    p.add_argument("--valid_ratio", type=float, default=0.025)

    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--batch_size_per_device", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--min_warmup_steps", type=int, default=1000)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--num_logging_steps_per_epoch", type=int, default=1000)
    p.add_argument("--num_eval_steps_per_epoch", type=int, default=10)
    p.add_argument("--num_save_steps_per_epoch", type=int, default=10)

    p.add_argument("--use_8bit", action="store_true")
    p.add_argument("--use_4bit", action="store_true")

    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    
    config = p.parse_args()

    return config


def get_now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def init_wandb(config):
    wandb.login()
    wandb.init(project="Meeting-Log-Summ", config=vars(config))
    wandb.run.name = config.model_name + '_' + get_now()        # Add timestamp to run name, because we want to compare different runs
    wandb.run.save()

    return wandb.run.name


def get_datasets(
        input_fn,
        prompt=\
"""### 회의록:
{dialogue}

### 회의 참석자:
{attendees}

### 주장하는 사람:
{assertion_person}

### 주장과 근거:
{assertion}

### 요약:
{summary}<|sep|><|endoftext|>""",
        valid_ratio=0.05,
        valid_fn=None,
    ):
    data = []

    # open jsonl file
    with open(input_fn, "r") as f:
        for line in f:
            js = json.loads(line)

            dialogue = js["dialogue"]
            summary = js["summary"]
            attendees = js["attendees"]
            assertion_person = js["assertion_person"]
            assertion = js["assertion"]

            summary = " ".join(summary).strip()
            attendees = "\n".join(attendees).strip()
            assertion_person = "\n".join(assertion_person).strip()
            assertion = "\n".join(assertion).strip()

            data.append(prompt.format(
                dialogue=dialogue,
                summary=summary,
                attendees=attendees,
                assertion_person=assertion_person,
                assertion=assertion,
            ))

    random.shuffle(data)
    print(f"Sample data:\n{data[0]}")

    train_data = data[:int(len(data) * (1 - valid_ratio))]
    valid_data = data[int(len(data) * (1 - valid_ratio)):]

    if valid_fn is not None:
        # check if directory is not exists
        if not os.path.exists(os.path.dirname(valid_fn)):
            os.makedirs(os.path.dirname(valid_fn))

        with open(valid_fn, "w") as f:
            for line in valid_data:
                f.write(line + "\n")

    train_dataset = datasets.Dataset.from_dict({"text": train_data})
    valid_dataset = datasets.Dataset.from_dict({"text": valid_data})

    return train_dataset, valid_dataset


def main(config):
    assert config.use_8bit ^ config.use_4bit, "You can only use one of 8bit and 4bit quantization."

    final_model_name = init_wandb(config)

    train_dataset, valid_dataset = get_datasets(
        config.input_fn,
        valid_ratio=config.valid_ratio,
        valid_fn=os.path.join(config.output_dir_path, final_model_name, "valid.txt")
    )

    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size:", len(valid_dataset))

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=config.max_length), batched=True)
    valid_dataset = valid_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=config.max_length), batched=True)

    # Get BitsAndBytesConfig for quantization
    if config.use_8bit:
        q_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    elif config.use_4bit:
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        q_config = None

    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        quantization_config=q_config,               # Use quantization, if necessary
        device_map="auto",                          # Let accelerate distributed model loading
        trust_remote_code=True,                     # Some models require this option
    )

    model.gradient_checkpointing_enable()           # Enable gradient checkpointing
    model = prepare_model_for_kbit_training(model)  # Prepare model for k-bit training

    l_config = LoraConfig(
        r=config.lora_r,                            # Set most important hyperparameters for Lora
        lora_alpha=config.lora_alpha,
        target_modules=["query_key_value"],         # Set target modules for Lora application
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, l_config)

    gpu_count = torch.cuda.device_count()
    total_batch_size = config.batch_size_per_device * gpu_count
    num_iterations_per_epoch = int((len(train_dataset) / total_batch_size) / config.gradient_accumulation_steps)
    logging_steps = max(10, int(num_iterations_per_epoch / config.num_logging_steps_per_epoch))
    eval_steps = max(10, int(num_iterations_per_epoch / config.num_eval_steps_per_epoch))
    save_steps = max(10, int(num_iterations_per_epoch / config.num_save_steps_per_epoch))
    warmup_steps = max(
        config.min_warmup_steps,
        num_iterations_per_epoch * config.num_train_epochs * config.warmup_ratio,
    )

    training_args = transformers.TrainingArguments(
        output_dir=os.path.join(config.output_dir_path, final_model_name),
        overwrite_output_dir=True,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,
        # fp16=True,
        # fp16_full_eval=True,
        half_precision_backend="auto",
        bf16=True,
        bf16_full_eval=True,
        report_to="wandb",
        optim="paged_adamw_8bit"
    )

    print(">> Training arguments:")
    print(training_args)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    trainer.save_model(os.path.join(config.output_dir_path, final_model_name))

    wandb.finish()


if __name__ == "__main__":
    config = get_config()
    main(config)
