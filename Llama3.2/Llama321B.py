import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# Enable model parallelism if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for model parallelism")
    model.parallelize()
else:
    print("Model parallelism not applied. Only one GPU detected.")
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Prompt
PROMPT = (
    "You're a model to select correct answers from the given questions and answer choices. "
    "The answer choices might look similar to each other but it's your job to figure out the correct one given the training you got.\n\n
    Here are some examples: 
{'id': 'SP-0', 'question': 'Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?', 'answer': 'Each daughter shares the same brother.', 'distractor1': 'Some daughters get married and have their own family.', 'distractor2': 'Some brothers were not loved by family and moved away.', 'distractor(unsure)': 'None of above.', 'label': 1, 'choice_list': ['Some daughters get married and have their own family.', 'Each daughter shares the same brother.', 'Some brothers were not loved by family and moved away.', 'None of above.'], 'choice_order': [1, 0, 2, 3]}\n
{'id': 'WP-131', 'question': 'What is a boxer’s favorite drink?', 'answer': 'Punch.', 'distractor1': 'Coke.', 'distractor2': 'Sprite.', 'distractor(unsure)': 'None of above.', 'label': 1, 'choice_list': ['Coke.', 'Punch.', 'Sprite.', 'None of above.'], 'choice_order': [1, 0, 2, 3]}\n
{'id': 'WP-119', 'question': 'What falls down but never breaks?', 'answer': 'Nightfall.', 'distractor1': 'Waterfall.', 'distractor2': 'Freefall.', 'distractor(unsure)': 'None of above.', 'label': 0, 'choice_list': ['Nightfall.', 'Waterfall.', 'Freefall.', 'None of above.'], 'choice_order': [0, 1, 2, 3]}\n
{'id': 'SP-136', 'question': 'A horse was tied to a rope 5 meters long and the horses food was 15 meters away from the horse. How did the horse reach the food?', 'answer': "The rope wasn't tied to anything so he could reach the food.", 'distractor1': 'The walls of the saloon retract or collapse inwards, creating more space for the horse to reach the food.', 'distractor2': 'The rope stretches proportionally, providing the extra length needed for the horse to reach the hay fifteen meters away.', 'distractor(unsure)': 'None of above.', 'label': 1, 'choice_list': ['The walls of the saloon retract or collapse inwards, creating more space for the horse to reach the food.', "The rope wasn't tied to anything so he could reach the food.", 'The rope stretches proportionally, providing the extra length needed for the horse to reach the hay fifteen meters away.', 'None of above.'], 'choice_order': [1, 0, 2, 3]}\n"
)

# Load datasets
train_data = np.load('../CombinedDatasets/All_train 1.npy', allow_pickle=True)
dev_data = np.load('../CombinedDatasets/All_dev 1.npy', allow_pickle=True)
test_data = np.load('../CombinedDatasets/All_test 1.npy', allow_pickle=True)

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


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

class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Callback to capture training logs."""
        if logs and "loss" in logs:
            step = state.global_step
            training_logs.append({
                "Learning Rate": args.learning_rate,
                "Weight Decay": args.weight_decay,
                "Step": step,
                "Training Loss": logs.get("loss", None),
                "Eval Loss": logs.get("eval_loss", None)
            })

# Training parameters
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)


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

        # Generate model prediction
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute cosine similarity with embeddings
        generated_embedding = embedder.encode(generated_text, convert_to_tensor=True)
        choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
        cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)

        # Choose the answer with the highest cosine similarity
        best_choice_index = torch.argmax(cosine_similarities).item()
        predicted_answer = choices[best_choice_index]

        # Check if the prediction matches the correct answer
        if predicted_answer == correct_answer:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy


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
            output_dir=f"./llama_lora_finetuned_lr{lr}_wd{wd}",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=10,
            save_steps=10,
            eval_steps=10,
            learning_rate=lr,
            weight_decay=wd,
            fp16=True,  # Enables mixed precision training
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
            callbacks=[LogCallback()]  # Use the class-based callback
        )

        torch.cuda.empty_cache()

        # Train the model
        trainer.train()

        # Save the best-performing model
        best_model_dir = f"./llama_lora_best_model_lr{lr}_wd{wd}"
        trainer.save_model(best_model_dir)

        # Reload the best-performing model
        best_model = AutoModelForCausalLM.from_pretrained(best_model_dir, torch_dtype=torch.float16).to(device)

        # Calculate accuracy on the test set using the reloaded best model
        test_accuracy = calculate_accuracy_with_embeddings(best_model, processed_test_data)
        results.append({"Learning Rate": lr, "Weight Decay": wd, "Accuracy": test_accuracy})



# Save training logs to CSV
def save_training_logs_to_csv(logs, filename="llama_lora_training_logs.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file,
                                fieldnames=["Learning Rate", "Weight Decay", "Step", "Training Loss", "Eval Loss"])
        writer.writeheader()
        writer.writerows(logs)


# Save results to CSV
def save_results_to_csv(results, filename="llama_lora_results.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Learning Rate", "Weight Decay", "Accuracy"])
        writer.writeheader()
        writer.writerows(results)


save_results_to_csv(results)
save_training_logs_to_csv(training_logs)
print(f"Results saved to llama_lora_results.csv and training logs saved to llama_lora_training_logs.csv")
