import torch
import transformers
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import numpy as np

# Load the tokenizer and model for LLaMA 3.2B
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-3.2b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-3.2b-chat-hf",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Hyperparameters for LoRA
CUTOFF_LEN = 256  # Context length
LORA_R = 4
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.2

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],  # LLaMA modules to target for LoRA
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# Load the training and dev datasets
dataWP = np.load('TestAutri/datasets-brainteasers/WP_train 1.npy', allow_pickle=True).tolist()
dataSP = np.load('TestAutri/datasets-brainteasers/SP_train 1.npy', allow_pickle=True).tolist()
data = dataWP + dataSP  # Combine datasets

devWP = np.load('TestAutri/datasets-brainteasers/WP_dev 1.npy', allow_pickle=True).tolist()
devSP = np.load('TestAutri/datasets-brainteasers/SP_dev 1.npy', allow_pickle=True).tolist()
dev_data = devWP + devSP  # Combine dev datasets

# Prepare inputs
def ParseQuestion(question):
    parsed_question = question['question'] + "\nChoose one of the following answers and give an explanation below the answer:\n"
    for i in question['choice_order']:
        parsed_question += question['choice_list'][i] + "\n"
    parsed_question += "The correct answer is {}\n".format(question['answer'])
    return parsed_question

def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length"
    )

# Prepare the training dataset
prompts = [ParseQuestion(item) for item in data]
tokenized_data = [tokenize(prompt) for prompt in prompts]
input_ids = [td['input_ids'] for td in tokenized_data]

dataset = Dataset.from_dict({'input_ids': input_ids})

# Prepare the dev dataset
dev_prompts = [ParseQuestion(item) for item in dev_data]
tokenized_dev_data = [tokenize(prompt) for prompt in dev_prompts]
dev_input_ids = [td['input_ids'] for td in tokenized_dev_data]

dev_dataset = Dataset.from_dict({'input_ids': dev_input_ids})

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.1,
    max_grad_norm=0.3,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    optim="adamw_torch",
    output_dir="llama-3.2b-lora-brainteasers",
    save_total_limit=2,
    report_to="none"  # Disable reporting to third-party services
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dev_dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

# Train the model
trainer.train()

print("Model fine-tuned and checkpoints saved to ./llama-3.2b-lora-brainteasers")
