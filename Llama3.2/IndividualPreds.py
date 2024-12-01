import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import csv
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load test dataset
test_data = np.load("/home/jawadkk/Brainteaser-GPT2/CombinedDatasets/All_test 1.npy", allow_pickle=True)

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
            'id': item['id'],  # Use actual ID from the dataset (e.g., 'SP-180')
            'text': question,
            'choices': choices,
            'correct_answer': choices[label]
        })
    return processed_data


processed_test_data = preprocess_data(test_data)


# Generate answers using the model
def generate_answer(model, tokenizer, question, choices):
    # Define the prompt dynamically based on the question and choices
    prompt = (
            "Answer the following question by selecting the most appropriate choice:\n"
            f"Question: {question}\nChoices:\n"
            + "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
            + "\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    explanation = "Based on context and semantic similarity, the chosen answer matches the question."
    return generated_text.split("Answer:")[-1].strip(), explanation


# Refine the generated answer using cosine similarity
def refine_prediction_with_similarity(generated_answer, choices):
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    best_index = torch.argmax(cosine_similarities).item()
    return choices[best_index]


# Evaluate the model on the test data
def evaluate_model(model, tokenizer, test_data, output_file):
    predictions = []
    correct_predictions = 0
    for item in test_data:
        question_id = item['id']
        question = item['text']
        choices = item['choices']
        correct_answer = item['correct_answer']

        # Generate and refine answer
        generated_answer, explanation = generate_answer(model, tokenizer, question, choices)
        refined_answer = refine_prediction_with_similarity(generated_answer, choices)

        # Debug information
        is_correct = "yes" if refined_answer == correct_answer else "no"
        predictions.append({
            "Question_ID": question_id,
            "Question_Text": question,
            "Choices": '; '.join(choices),
            "Generated Answer": generated_answer,
            "Refined Answer": refined_answer,
            "Correct Answer": correct_answer,
            "Predicted == Correct": is_correct,
            "Explanation": explanation
        })
        if is_correct == "yes":
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save predictions to a CSV
    save_predictions_to_csv(predictions, output_file)
    return accuracy


# Save predictions to CSV
def save_predictions_to_csv(predictions, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question_ID", "Question_Text", "Choices",
                                                  "Generated Answer", "Refined Answer",
                                                  "Correct Answer", "Predicted == Correct", "Explanation"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Predictions saved to {filename}")


# Evaluate all combinations
def evaluate_all_combinations(processed_test_data, learning_rates, weight_decays,
                              base_model_dir="/home/jawadkk/Brainteaser-GPT2/Llama3.2/"):
    for lr in learning_rates:
        for wd in weight_decays:
            model_id = f"llama_lora_finetuned_lr{lr}_wd{wd}"
            model_path = os.path.join(base_model_dir, model_id)
            output_file = f"Results/{model_id}_results.csv"
            os.makedirs("Results", exist_ok=True)
            try:
                # Debug: Print model path
                print(f"Loading model from {model_path}")

                # Load the fine-tuned model
                model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id  # Avoid warning during generation

                # Evaluate the model
                accuracy = evaluate_model(model, tokenizer, processed_test_data, output_file)
                print(f"Model {model_id} Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"Error evaluating {model_id}: {e}")


# Run evaluation
evaluate_all_combinations(processed_test_data, learning_rates, weight_decays)
