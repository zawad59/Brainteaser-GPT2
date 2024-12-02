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
test_data = np.load("/home/jawadkk/Brainteaser-GPT2/CombinedDatasets/All_test 1.npy", allow_pickle=True)

# Define learning rates and weight decays
learning_rates = [0.0001]
weight_decays = [0.005]

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
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    explanation = "Based on context and semantic similarity, the chosen answer matches the question."
    return generated_text.split("Answer:")[-1].strip(), explanation

def refine_prediction_with_similarity(generated_answer, choices):
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    best_index = torch.argmax(cosine_similarities).item()
    return choices[best_index]

# Evaluate the model on the test data
def evaluate_model(model, tokenizer, test_data, output_file):
    results = []
    for item in test_data:
        question_id = item['id']
        question = item['text']
        choices = item['choices']
        correct_answer = item['correct_answer']
        generated_answer, explanation = generate_answer(model, tokenizer, question, choices)
        refined_answer = refine_prediction_with_similarity(generated_answer, choices)
        prediction_match = "yes" if generated_answer == correct_answer else "no"
        results.append({
            "Question_ID": question_id,
            "Question_Text": question,
            "Answer_Choices": "; ".join(choices),
            "Model_Predicted_Answer": refined_answer,
            "Actual_Answer": correct_answer,
            "Predicted == Actual": prediction_match,
            "Explanation": explanation
        })

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Evaluate all combinations
def evaluate_all_combinations(processed_test_data, learning_rates, weight_decays, base_model_dir="/home/jawadkk/Brainteaser-GPT2/Llama3.2/"):
    for lr in learning_rates:
        for wd in weight_decays:
            model_id = f"llama_lora_finetuned_lr{lr}_wd{wd}"
            model_path = os.path.join(base_model_dir, model_id)
            print(model_path)
            output_file = f"Results/{model_id}_results.csv"
            os.makedirs("Results", exist_ok=True)
            try:
                # Load the fine-tuned model
                model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id  # Avoid warning during generation

                # Evaluate the model
                evaluate_model(model, tokenizer, processed_test_data, output_file)
            except Exception as e:
                print(f"Error evaluating {model_id}: {e}")

# Run evaluation
evaluate_all_combinations(processed_test_data, learning_rates, weight_decays)
