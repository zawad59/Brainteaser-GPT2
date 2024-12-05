import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
import gc
from trl import SFTTrainer, setup_chat_format
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer and model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
).to(device)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Set constants
CUTOFF_LEN = 256
LORA_R = 4
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1

# LoRA fine-tuning configuration
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["query_proj", "value_proj"],  # Target MoE layers
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Function to preprocess data
def preprocess_question(item):
    parsed_question = (
        f"Question: {item['question']}\n"
        "Choose the correct answer from the following options:\n"
    )
    for i in item['choice_order']:
        parsed_question += f"{item['choice_list'][i]}\n"
    parsed_question += f"The correct answer is {item['answer']}.\n"
    return parsed_question

def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors="pt"
    )

# Load and preprocess datasets
print("Loading datasets...")
train_data = np.load('../CombinedDatasets/All_train 1.npy', allow_pickle=True).tolist()
dev_data = np.load('../CombinedDatasets/All_dev 1.npy', allow_pickle=True).tolist()

train_prompts = [preprocess_question(item) for item in train_data]
dev_prompts = [preprocess_question(item) for item in dev_data]

tokenized_train_data = [tokenize(prompt) for prompt in train_prompts]
tokenized_dev_data = [tokenize(prompt) for prompt in dev_prompts]

train_input_ids = [td["input_ids"] for td in tokenized_train_data]
dev_input_ids = [td["input_ids"] for td in tokenized_dev_data]

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"input_ids": train_input_ids})
dev_dataset = Dataset.from_dict({"input_ids": dev_input_ids})

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama_lora_finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.2,
    max_grad_norm=0.3,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    optim="adamw_torch"
)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    max_seq_length= 512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    packing= False,
)

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model
output_dir = "./LlamaFinetuned"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving the fine-tuned model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Clear memory
del trainer, model
torch.cuda.empty_cache()
gc.collect()

print(f"Fine-tuning complete. Model saved to '{output_dir}'")
