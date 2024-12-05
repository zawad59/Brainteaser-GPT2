import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import gc

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
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Function to preprocess data
def ParseQuestion(question):
    """
    Parses a question dictionary into a structured text prompt.
    """
    parsed_question = (
        f"Question: {question['question']}\n"
        "Choose one of the following answers and give an explanation below the answer:\n"
    )
    for i in question['choice_order']:
        parsed_question += f"- {question['choice_list'][i]}\n"
    parsed_question += f"The correct answer is: {question['answer']}\n"
    return parsed_question

def tokenize(prompt):
    """
    Tokenizes a given prompt using the tokenizer.
    """
    encoded = tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors="pt"
    )
    return {"input_ids": encoded["input_ids"][0], "attention_mask": encoded["attention_mask"][0]}

# Load and preprocess datasets
print("Loading datasets...")
train_data = np.load('../CombinedDatasets/All_train 1.npy', allow_pickle=True).tolist()
dev_data = np.load('../CombinedDatasets/All_dev 1.npy', allow_pickle=True).tolist()

train_prompts = [ParseQuestion(item) for item in train_data]
dev_prompts = [ParseQuestion(item) for item in dev_data]

tokenized_train_data = [tokenize(prompt) for prompt in train_prompts]
tokenized_dev_data = [tokenize(prompt) for prompt in dev_prompts]

train_input_ids = [td["input_ids"] for td in tokenized_train_data]
train_attention_masks = [td["attention_mask"] for td in tokenized_train_data]

dev_input_ids = [td["input_ids"] for td in tokenized_dev_data]
dev_attention_masks = [td["attention_mask"] for td in tokenized_dev_data]

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"input_ids": train_input_ids, "attention_mask": train_attention_masks})
dev_dataset = Dataset.from_dict({"input_ids": dev_input_ids, "attention_mask": dev_attention_masks})

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

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
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
