import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from datasets import Dataset as HFDataset
import csv
from transformers import TrainerCallback

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load datasets
train_data = np.load('../CombinedDatasets/All_train 1.npy', allow_pickle=True)
dev_data = np.load('../CombinedDatasets/All_dev 1.npy', allow_pickle=True)
test_data = np.load('../CombinedDatasets/All_test 1.npy', allow_pickle=True)

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


class LogCallback(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.logs.append({
                "Step": state.global_step,
                "Train Loss": logs.get("loss"),
                "Validation Loss": logs.get("eval_loss"),
            })


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
    "{'id': 'WP-119', 'question': 'What falls down but never breaks?', 'answer': 'Nightfall.', 'distractor1': 'Waterfall.', 'distractor2': 'Freefall.', "
    "'distractor(unsure)': 'None of above.', 'label': 0, 'choice_list': ['Nightfall.', 'Waterfall.', 'Freefall.', 'None of above.'], 'choice_order': [0, 1, 2, 3]}\n"
    "{'id': 'SP-136', 'question': 'A horse was tied to a rope 5 meters long and the horses food was 15 meters away from the horse. How did the horse reach the food?', "
    "'answer': \"The rope wasn't tied to anything so he could reach the food.\", 'distractor1': 'The walls of the saloon retract or collapse inwards, creating more space for the horse to reach the food.', "
    "'distractor2': 'The rope stretches proportionally, providing the extra length needed for the horse to reach the hay fifteen meters away.', "
    "'distractor(unsure)': 'None of above.', 'label': 1, 'choice_list': ['The walls of the saloon retract or collapse inwards, creating more space for the horse to reach the food.', "
    "\"The rope wasn't tied to anything so he could reach the food.\", 'The rope stretches proportionally, providing the extra length needed for the horse to reach the hay fifteen meters away.', 'None of above.'], "
    "'choice_order': [1, 0, 2, 3]}\n"
)


# Preprocess the data
def preprocess_llama_data(data):
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
            f"{PROMPT}Question: {cleaned_question}\n"
            f"Choices: {', '.join(choices)}\n"
            f"Answer: {correct_answer}\n\n"
        )
        processed_data.append({'text': training_text, 'choices': choices, 'label': item['label']})
    return processed_data


# Preprocess the datasets
processed_train_data = preprocess_llama_data(train_data)
processed_dev_data = preprocess_llama_data(dev_data)
processed_test_data = preprocess_llama_data(test_data)


# Convert to Hugging Face Dataset
def create_hf_dataset(processed_data):
    return HFDataset.from_list(processed_data)


def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# Tokenize datasets
train_dataset = create_hf_dataset(processed_train_data).map(tokenize_function, batched=True,
                                                            remove_columns=["text", "choices", "label"])
dev_dataset = create_hf_dataset(processed_dev_data).map(tokenize_function, batched=True,
                                                        remove_columns=["text", "choices", "label"])


# Function to calculate accuracy using embeddings
def calculate_accuracy_with_embeddings(model, test_data):
    correct_predictions = 0
    total_predictions = len(test_data)

    for example in test_data:
        # Extract question and choices
        question = example['text']
        choices = example['choices']
        correct_label = example['label']
        correct_answer = choices[correct_label]

        # Tokenize input
        inputs = tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Generate predictions
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compare with choices using cosine similarity
        generated_embedding = embedder.encode(generated_text, convert_to_tensor=True)
        choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
        cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)

        # Choose the answer with the highest cosine similarity
        best_choice_index = torch.argmax(cosine_similarities).item()
        predicted_answer = choices[best_choice_index]

        # Check if the prediction matches the correct answer
        if predicted_answer == correct_answer:
            correct_predictions += 1

    # Return accuracy
    return correct_predictions / total_predictions


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


# Save training logs
def save_training_logs_to_csv(logs, filename="llama_lora_training_logs.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Step", "Train Loss", "Validation Loss"])
        writer.writeheader()
        writer.writerows(logs)


for lr in learning_rates:
    for wd in weight_decays:
        # Prepare the base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16).to(device)
        model = get_peft_model(base_model, lora_config)

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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[LogCallback()]
        )

        # Train the model
        trainer.train()

        # Save training logs
        save_training_logs_to_csv(trainer.state.log_history, filename="llama_lora_training_logs.csv")

        # Save the adapter
        adapter_dir = f"./llama_lora_best_model_lr{lr}_wd{wd}"
        model.save_pretrained(adapter_dir)

        # Reload base model and adapter
        model_with_adapter = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

        # Calculate accuracy
        test_accuracy = calculate_accuracy_with_embeddings(model_with_adapter, processed_test_data)
        results.append({"Learning Rate": lr, "Weight Decay": wd, "Accuracy": test_accuracy})

        # Clear GPU memory
        del model, model_with_adapter, trainer
        torch.cuda.empty_cache()
        gc.collect()


# Save results to CSV
def save_results_to_csv(results, filename="llama_lora_results.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Learning Rate", "Weight Decay", "Accuracy"])
        writer.writeheader()
        writer.writerows(results)


save_results_to_csv(results)
print(f"Results saved to llama_lora_results.csv")
