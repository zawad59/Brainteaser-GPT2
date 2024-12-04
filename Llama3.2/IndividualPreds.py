import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import csv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load test dataset
test_data = np.load("/home/jawadkk/Brainteaser-GPT2/CombinedDatasets/All_test 1.npy", allow_pickle=True)

# Preprocess the test dataset
def preprocess_data(data):
    processed_data = []
    for item in data:
        question = item['question']
        choices = item['choice_list']
        label = item['label']
        processed_data.append({
            'id': item['id'],
            'text': question,
            'choices': choices,
            'correct_answer': choices[label]
        })
    return processed_data

processed_test_data = preprocess_data(test_data)

# Generate answers using the model
def generate_answer(model, tokenizer, question, choices):
   prompt = (
        "Using the fine-tuned training models, learn to generate responses that are accurate and aligned with the examples provided. "
        "Based on the given examples, generate responses to the following question without producing gibberish:\n\n"
        "Example 1:\n"
        "{'id': 'WP-131_CR', 'question': \"What is a gardener's favorite type of music?\", 'answer': 'Rock.', "
        "'distractor1': 'Jazz.', 'distractor2': 'Blue.', 'distractor(unsure)': 'None of above.', "
        "'label': 0, 'choice_list': ['Rock.', 'Blue.', 'Jazz.', 'None of above.'], 'choice_order': [0, 2, 1, 3]}\n\n"
        "Example 2:\n"
        "{'id': 'SP-149_SR', 'question': 'What question can someone ask all day and receive radically different responses, "
        "yet all of them might be correct?', 'answer': 'What time is it?', 'distractor1': \"What's the square root of 16?\", "
        "'distractor2': 'What is the result of 5317 by 9321.', 'distractor(unsure)': 'None of above.', "
        "'label': 1, 'choice_list': [\"What's the square root of 16?\", 'What time is it?', 'What is the result of 5317 by 9321.', "
        "'None of above.'], 'choice_order': [1, 0, 2, 3]}\n\n"
        f"Now answer this question:\n"
        f"Question: {question}\n"
        "Choices:\n" +
        "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)]) +
        "
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_part = generated_text.split("Answer (choose the number only):")[-1].strip()
    return answer_part.split("\n")[0].strip()  # Return the raw choice number

# Refine prediction using cosine similarity
def refine_prediction_with_similarity(generated_answer, choices):
    try:
        # Check if generated answer is a valid choice number
        answer_index = int(generated_answer) - 1
        if 0 <= answer_index < len(choices):
            return choices[answer_index]
    except ValueError:
        pass  # Continue to similarity matching if not valid

    # Fall back to semantic similarity
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    best_index = torch.argmax(cosine_similarities).item()
    return choices[best_index]

# Evaluate the model
def evaluate_model(model, tokenizer, test_data, output_file):
    predictions = []
    correct_predictions = 0
    for item in test_data:
        question_id = item['id']
        question = item['text']
        choices = item['choices']
        correct_answer = item['correct_answer']

        generated_answer = generate_answer(model, tokenizer, question, choices)
        refined_answer = refine_prediction_with_similarity(generated_answer, choices)

        is_correct = refined_answer == correct_answer
        predictions.append({
            "Question_ID": question_id,
            "Question_Text": question,
            "Choices": '; '.join(choices),
            "Generated Answer": generated_answer,
            "Refined Answer": refined_answer,
            "Correct Answer": correct_answer,
            "Predicted == Correct": "yes" if is_correct else "no"
        })
        if is_correct:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    print(f"Test Accuracy: {accuracy:.4f}")
    save_predictions_to_csv(predictions, output_file)
    return accuracy

# Define learning rates and weight decays
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
weight_decays = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

# Save predictions to CSV
def save_predictions_to_csv(predictions, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question_ID", "Question_Text", "Choices",
                                                  "Generated Answer", "Refined Answer",
                                                  "Correct Answer", "Predicted == Correct"])
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
            output_file = f"ResultsFewShot/{model_id}_results.csv"
            os.makedirs("ResultsFewShot", exist_ok=True)
            try:
                print(f"Loading model from {model_path}")
                model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

                accuracy = evaluate_model(model, tokenizer, processed_test_data, output_file)
                print(f"Model {model_id} Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"Error evaluating {model_id}: {e}")

# Run evaluation
evaluate_all_combinations(processed_test_data, learning_rates, weight_decays)
