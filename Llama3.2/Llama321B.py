import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from datasets import Dataset as HFDataset
import csv

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer and load the base model
base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
# Ensure pad token is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Use the ID of eos_token


# Load datasets
train_data = np.load('../CombinedDatasets/All_train 1.npy', allow_pickle=True)
dev_data = np.load('../CombinedDatasets/All_dev 1.npy', allow_pickle=True)
test_data = np.load('../CombinedDatasets/All_test 1.npy', allow_pickle=True)

# Preprocess the SP dataset
def preprocess_sp_data(data):
    processed_data = []
    for item in data:
        question = item['question']
        choices = item['choice_list']
        correct_answer = choices[item['label']]
        cleaned_question = question.lower()  # Preserve original context

        # Create training text
        training_text = (
                f"Question: {cleaned_question}\n"
                f"Choices:\n" + "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:"
        )
        processed_data.append({'text': training_text, 'choices': choices, 'label': item['label']})
    return processed_data

# Process and save datasets if not already processed
if not os.path.exists("processed_train_dataset.pt") or not os.path.exists("processed_dev_dataset.pt"):
    print("Processing datasets...")
    processed_train_data = preprocess_sp_data(train_data)
    processed_dev_data = preprocess_sp_data(dev_data)
    processed_test_data = preprocess_sp_data(test_data)

    # Convert to Hugging Face Dataset
    train_dataset = HFDataset.from_list(processed_train_data)
    dev_dataset = HFDataset.from_list(processed_dev_data)
    test_dataset = HFDataset.from_list(processed_test_data)

    # Tokenize datasets
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True,
                                                remove_columns=["text", "choices", "label"])
    tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=["text", "choices", "label"])

    # Save tokenized datasets
    torch.save(tokenized_train_dataset, "processed_train_dataset.pt")
    torch.save(tokenized_dev_dataset, "processed_dev_dataset.pt")
    print("Datasets processed and saved.")
else:
    print("Loading preprocessed datasets...")
    tokenized_train_dataset = torch.load("processed_train_dataset.pt")
    tokenized_dev_dataset = torch.load("processed_dev_dataset.pt")

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# LoRA fine-tuning configuration
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# Prepare model for LoRA fine-tuning
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16).to(device)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Fine-tuning configurations
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
weight_decays = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

# CSV file to log training details
log_csv_file = "llama_training_logs.csv"
os.makedirs("Results", exist_ok=True)

# Initialize CSV file with headers
with open(f"Results/{log_csv_file}", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Model_ID", "loss", "grad_norm", "learning_rate", "epoch", "step",
                                              "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"])
    writer.writeheader()

# Train the model with different hyperparameter combinations
for lr in learning_rates:
    for wd in weight_decays:
        model_id = f"Llama3.2_3Bparam_lr{lr}_wd{wd}"

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"./llama_lora_finetuned_lr{lr}_wd{wd}",
            num_train_epochs=5,
            per_device_train_batch_size=8,  # Reduced batch size
            per_device_eval_batch_size=8,  # Reduced eval batch size
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=10,
            save_steps=10,
            eval_steps=10,
            gradient_accumulation_steps=2,  # Simulate larger batches
            learning_rate=lr,
            weight_decay=wd,
            fp16=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none"
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

        # Save the fine-tuned model
        model.save_pretrained(f"./llama_lora_finetuned_lr{lr}_wd{wd}")

        # Clear memory
        del trainer, model
        torch.cuda.empty_cache()
        model = get_peft_model(base_model, lora_config)  # Reload the base model with LoRA
