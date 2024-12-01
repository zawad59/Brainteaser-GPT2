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

# Initialize Phi35 Model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
print(model)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Define prompt
PROMPT = (
    "You're a model to select correct answers from the given questions and answer choices. "
    "The answer choices might look similar to each other but it's your job to figure out the correct one given the training you got.\n\n"
    "Here are some examples:\n"
    "{'id': 'SP-0', 'question': 'Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?', "
    "'answer': 'Each daughter shares the same brother.', 'distractor1': 'Some daughters get married and have their own family.', "
    "'distractor2': 'Some brothers were not loved by family and moved away.', 'distractor(unsure)': 'None of above.', 'label': 1, "
    "'choice_list': ['Some daughters get married and have their own family.', 'Each daughter shares the same brother.', 'Some brothers were not loved by family and moved away.', 'None of above.'], 'choice_order': [1, 0, 2, 3]}\n"
    "{'id': 'WP-131', 'question': 'What is a boxer’s favorite drink?', 'answer': 'Punch.', 'distractor1': 'Coke.', 'distractor2': 'Sprite.', "
    "'distractor(unsure)': 'None of above.', 'label': 1, 'choice_list': ['Coke.', 'Punch.', 'Sprite.', 'None of above.'], 'choice_order': [1, 0, 2, 3]}\n"
)

# Load datasets
train_data = np.load('../CombinedDatasets/All_train 1.npy', allow_pickle=True)
dev_data = np.load('../CombinedDatasets/All_dev 1.npy', allow_pickle=True)

# Function to preprocess and tokenize data
def preprocess_and_tokenize(data):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    dataset = HFDataset.from_list([
        {
            "text": f"{PROMPT}Question: {item['question']}\nChoices: {', '.join(item['choice_list'])}\nAnswer: ",
            "label": item["label"],
            "choice_list": item["choice_list"]
        }
        for item in data
    ])
    return dataset.map(
        lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256),
        batched=True,
        remove_columns=["text", "label", "choice_list"]
    )

# Preprocess and tokenize datasets
print("Processing and tokenizing datasets...")
tokenized_train_dataset = preprocess_and_tokenize(train_data)
tokenized_dev_dataset = preprocess_and_tokenize(dev_data)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# LoRA fine-tuning configuration
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Fine-tuning configurations
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
weight_decays = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

# CSV file to log training details
log_csv_file = "phi35_training_logs.csv"
os.makedirs("Results", exist_ok=True)

# Initialize CSV file with headers
with open(f"Results/{log_csv_file}", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Model_ID", "loss", "grad_norm", "learning_rate", "epoch", "step",
                                              "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"])
    writer.writeheader()

# Train the model with different hyperparameter combinations
for lr in learning_rates:
    for wd in weight_decays:
        model_id = f"Phi35_lr{lr}_wd{wd}"

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"./phi35_finetuned_lr{lr}_wd{wd}",
            num_train_epochs=5,
            per_device_train_batch_size=16,  # Adjusted batch size for large model
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=10,
            save_steps=10,
            eval_steps=10,
            learning_rate=lr,
            weight_decay=wd
        )

        # Define custom Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_dev_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        # Train and log details
        trainer.train()

        # Log details to CSV
        with open(f"Results/{log_csv_file}", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["Model_ID", "loss", "grad_norm", "learning_rate", "epoch", "step",
                                                      "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"])
            for log in trainer.state.log_history:
                log_row = {
                    "Model_ID": model_id,
                    "loss": log.get("loss", "N/A"),
                    "grad_norm": log.get("grad_norm", "N/A"),
                    "learning_rate": log.get("learning_rate", "N/A"),
                    "epoch": log.get("epoch", "N/A"),
                    "step": log.get("step", "N/A"),
                    "eval_loss": log.get("eval_loss", "N/A"),
                    "eval_runtime": log.get("eval_runtime", "N/A"),
                    "eval_samples_per_second": log.get("eval_samples_per_second", "N/A"),
                    "eval_steps_per_second": log.get("eval_steps_per_second", "N/A"),
                }
                writer.writerow(log_row)

        # Save the fine-tuned model
        trainer.save_model(f"./phi35_finetuned_lr{lr}_wd{wd}")

print(f"Training logs saved to Results/{log_csv_file}")
