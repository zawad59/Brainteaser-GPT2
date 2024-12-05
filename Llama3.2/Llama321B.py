import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
import gc

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Define the base prompt
PROMPT = (
    "You're a model to select correct answers from the given questions and answer choices. "
    "The answer choices might look similar to each other but it's your job to figure out the correct one given the training you got.\n\n"
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Answer: {answer}\n"
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
            "text": PROMPT.format(
                question=item['question'],
                choices=', '.join(item['choice_list']),
                answer=item['answer']
            ),
        }
        for item in data
    ])
    return dataset.map(
        lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256),
        batched=True,
        remove_columns=["text"]
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
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama_lora_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=10,
    save_steps=10,
    eval_steps=10,
    learning_rate=0.0001,
    weight_decay=0.1,
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

# Train the model
print("Starting fine-tuning...")
trainer.train()

# Directory to save the fine-tuned model
output_dir = "./LlamaFinetuned"
os.makedirs(output_dir, exist_ok=True)

# Save the fine-tuned model
print(f"Saving the fine-tuned model to {output_dir}...")
trainer.save_model(output_dir)

# Clear memory
del trainer, model
torch.cuda.empty_cache()
gc.collect()

print(f"Fine-tuning complete. Model saved to '{output_dir}'")
