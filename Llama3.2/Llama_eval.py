import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load test dataset
test_data = np.load("All_test 1.npy", allow_pickle=True)

# Define learning rates and weight decays
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
weight_decays = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

# Preprocess the test dataset
def preprocess_data(data):
    processed_data = []
    for item in data:
        question = item['question']
        choices = item['choice_list']
        label = item['label']
        processed_data.append({
            'text': question,
            'choices': choices,
            'correct_answer': choices[label]
        })
    return processed_data

processed_test_data = preprocess_data(test_data)

# Define the prompt
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

# Generate answers using the model
def generate_answer(model, tokenizer, question, choices):
    prompt = (
        PROMPT
        + f"\n\nQuestion: {question}\nChoices:\n"
        + "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
        + "\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("Answer:")[-1].strip()

# Refine the generated answer using cosine similarity
def refine_prediction_with_similarity(embedder, generated_answer, choices):
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    best_index = torch.argmax(cosine_similarities).item()
    return choices[best_index]

# Evaluate the model on the test data
def evaluate_model(model, tokenizer, test_data):
    correct_count = 0
    for item in test_data:
        question = item['text']
        choices = item['choices']
        correct_answer = item['correct_answer']
        generated_answer = generate_answer(model, tokenizer, question, choices)
        refined_answer = refine_prediction_with_similarity(embedder, generated_answer, choices)
        if refined_answer == correct_answer:
            correct_count += 1
    return correct_count / len(test_data)

# Evaluate all combinations
def evaluate_all_combinations(processed_test_data, learning_rates, weight_decays, base_model_dir="../"):
    results = []
    for lr in learning_rates:
        for wd in weight_decays:
            model_id = f"llama_lora_finetuned_lr{lr}_wd{wd}"
            model_path = os.path.join(base_model_dir, model_id)
            try:
                # Load the fine-tuned model
                model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                tokenizer.pad_token = tokenizer.eos_token
                
                # Evaluate the model
                accuracy = evaluate_model(model, tokenizer, processed_test_data)
                
                # Log results
                results.append({
                    "Model ID": model_id,
                    "Learning Rate": lr,
                    "Weight Decay": wd,
                    "Accuracy": accuracy
                })
                print(f"Evaluated {model_id}: Accuracy = {accuracy:.4f}")
            except Exception as e:
                print(f"Error evaluating {model_id}: {e}")
                results.append({
                    "Model ID": model_id,
                    "Learning Rate": lr,
                    "Weight Decay": wd,
                    "Accuracy": None
                })
    return results

# Run evaluation and save results
results = evaluate_all_combinations(processed_test_data, learning_rates, weight_decays)
results_df = pd.DataFrame(results)
results_df.to_csv("Llama_finetuning_results_with_prompt.csv", index=False)
print("Results saved to Llama_finetuning_results_with_prompt.csv")
