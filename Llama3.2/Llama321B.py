import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
import csv
import gc

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.config.pad_token_id = tokenizer.pad_token_id

# Load datasets
train_data = np.load('../CombinedDatasets/All_train 1.npy', allow_pickle=True)
dev_data = np.load('../CombinedDatasets/All_dev 1.npy', allow_pickle=True)

# Preprocess and tokenize datasets
def preprocess_and_tokenize(data):
    dataset = HFDataset.from_list(data)
    def tokenize_function(examples):
        tokens = tokenizer(examples["question"], padding='max_length', truncation=True, max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    return dataset.map(tokenize_function, batched=True, remove_columns=["question", "choice_list", "label"])

tokenized_train_dataset = preprocess_and_tokenize(train_data)
tokenized_dev_dataset = preprocess_and_tokenize(dev_data)

# LoRA configuration
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# Training configurations
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
weight_decays = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
log_csv_file = "Results/llama3_2_training_logs.csv"

# Initialize CSV file
def initialize_csv_file(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Model_ID", "loss", "grad_norm", "learning_rate", "epoch", "step", 
            "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"
        ])
        writer.writeheader()

# Append logs to CSV
def append_logs_to_csv(logs, filename, model_id):
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Model_ID", "loss", "grad_norm", "learning_rate", "epoch", "step", 
            "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"
        ])
        for log in logs:
            log_row = {
                "Model_ID": model_id,
                "loss": log.get("loss"),
                "grad_norm": log.get("grad_norm"),
                "learning_rate": log.get("learning_rate"),
                "epoch": log.get("epoch"),
                "step": log.get("step"),
                "eval_loss": log.get("eval_loss"),
                "eval_runtime": log.get("eval_runtime"),
                "eval_samples_per_second": log.get("eval_samples_per_second"),
                "eval_steps_per_second": log.get("eval_steps_per_second"),
            }
            writer.writerow(log_row)

# Initialize CSV file for logs
initialize_csv_file(log_csv_file)

# Fine-tune model with different hyperparameter combinations
for lr in learning_rates:
    for wd in weight_decays:
        model_id = f"Llama3.2_3Bparam_lr{lr}_wd{wd}"
        print(f"Training {model_id}...")

        # Prepare the model with LoRA
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=f"./llama3_2_lora_finetuned_lr{lr}_wd{wd}",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=10,
            save_steps=10,
            eval_steps=10,
            learning_rate=lr,
            weight_decay=wd,
            fp16=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_dev_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            tokenizer=tokenizer
        )

        # Train the model
        trainer.train()

        # Append logs to CSV
        append_logs_to_csv(trainer.state.log_history, log_csv_file, model_id)

        # Save fine-tuned model
        model.save_pretrained(f"./llama3_2_lora_best_model_lr{lr}_wd{wd}")

        # Clear memory
        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()

        # Reload base model for next run
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

print(f"Training complete. Logs saved to {log_csv_file}")
