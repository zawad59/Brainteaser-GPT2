import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
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

# Initialize models
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Load datasets
train_data = np.load('../CombinedDatasets/All_train 1.npy', allow_pickle=True)
dev_data = np.load('../CombinedDatasets/All_dev 1.npy', allow_pickle=True)
test_data = np.load('../CombinedDatasets/All_test 1.npy', allow_pickle=True)

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess the data
def preprocess_gpt2_data(data):
    processed_data = []
    for item in data:
        question = item['question']
        choices = item['choice_list']
        correct_answer = choices[item['label']]

        sentences = sent_tokenize(question)
        cleaned_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
            cleaned_sentence = ' '.join(filtered_words)
            cleaned_sentences.append(cleaned_sentence)

        cleaned_question = ' '.join(cleaned_sentences)
        training_text = (
            f"Question: {cleaned_question}\n"
            f"Choices: {', '.join(choices)}\n"
            f"Answer: {correct_answer}\n\n"
        )
        processed_data.append({'text': training_text, 'choices': choices, 'label': item['label']})
    return processed_data

# Preprocess the datasets
processed_train_data = preprocess_gpt2_data(train_data)
processed_dev_data = preprocess_gpt2_data(dev_data)
processed_test_data = preprocess_gpt2_data(test_data)

# Convert to Hugging Face Dataset
def create_hf_dataset(processed_data):
    return HFDataset.from_list(processed_data)

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Tokenize datasets
train_dataset = create_hf_dataset(processed_train_data).map(tokenize_function, batched=True, remove_columns=["text", "choices", "label"])
dev_dataset = create_hf_dataset(processed_dev_data).map(tokenize_function, batched=True, remove_columns=["text", "choices", "label"])

# Training parameters
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# Fine-tuning configurations
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
weight_decays = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
results = []
training_logs = []

# Iterate through learning rate and weight decay combinations
for lr in learning_rates:
    for wd in weight_decays:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=f"./gpt2_lora_finetuned_lr{lr}_wd{wd}",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="steps",  # Matches save_strategy
            save_strategy="steps",        # Updated to match evaluation_strategy
            logging_strategy="steps",
            logging_steps=10,
            save_steps=10,                # Saves every 10 steps
            eval_steps=10,                # Evaluates every 10 steps
            learning_rate=lr,
            weight_decay=wd,
            fp16=torch.cuda.is_available(),
            save_total_limit=1,
            load_best_model_at_end=True,  # Ensures best model is loaded
            report_to="none"
        )

        def log_callback(trainer_state, trainer_control, logs=None):
            """Callback to capture training logs."""
            if logs and "loss" in logs:
                step = trainer_state.global_step
                training_logs.append({
                    "Learning Rate": lr,
                    "Weight Decay": wd,
                    "Step": step,
                    "Training Loss": logs.get("loss", None),
                    "Eval Loss": logs.get("eval_loss", None)
                })

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            tokenizer=tokenizer,
            callbacks=[log_callback]
        )

        # Train and evaluate
        trainer.train()
        trainer.save_model(f"./gpt2_lora_best_model_lr{lr}_wd{wd}")

        # Evaluate on the dev set
        eval_results = trainer.evaluate(dev_dataset)
        test_loss = eval_results["eval_loss"]
        results.append({"Learning Rate": lr, "Weight Decay": wd, "Accuracy": -test_loss})

# Save training logs to CSV
def save_training_logs_to_csv(logs, filename="gpt2_lora_training_logs.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Learning Rate", "Weight Decay", "Step", "Training Loss", "Eval Loss"])
        writer.writeheader()
        writer.writerows(logs)

# Save results to CSV
def save_results_to_csv(results, filename="gpt2_lora_results.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Learning Rate", "Weight Decay", "Accuracy"])
        writer.writeheader()
        writer.writerows(results)

save_results_to_csv(results)
save_training_logs_to_csv(training_logs)
print(f"Results saved to gpt2_lora_results.csv and training logs saved to gpt2_lora_training_logs.csv")
