import os
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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

# Generate answers using the GPT-2 model
def generate_answer(model, tokenizer, question, choices):
    prompt = (
        "Below are examples of questions with their answers. Use these examples to guide your response. "
        "Choose the correct option from the provided choices based on the question:\n\n"
        "Example 1:\n"
        "Question: What is a gardener's favorite type of music?\n"
        "Choices:\n"
        "1. Rock.\n"
        "2. Blue.\n"
        "3. Jazz.\n"
        "4. None of the above.\n"
        "Answer: 1\n\n"
        "Example 2:\n"
        "Question: What question can someone ask all day and receive radically different responses, "
        "yet all of them might be correct?\n"
        "Choices:\n"
        "1. What time is it?\n"
        "2. What's the square root of 16?\n"
        "3. What is the result of 5317 by 9321?\n"
        "4. None of the above.\n"
        "Answer: 1\n\n"
        f"Now answer this question:\n"
        f"Question: {question}\n"
        "Choices:\n" +
        "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)]) +
        "\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.5,
        top_p=0.9,
        top_k=50,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_part = generated_text.split("Answer:")[-1].strip()
    return answer_part.split("\n")[0].strip()  # Return the extracted answer

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

# Evaluate the GPT-2 model
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
learning_rates = [0.1]
weight_decays = [0.05]

# Save predictions to CSV
def save_predictions_to_csv(predictions, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question_ID", "Question_Text", "Choices",
                                                  "Generated Answer", "Refined Answer",
                                                  "Correct Answer", "Predicted == Correct"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Predictions saved to {filename}")

# Evaluate all combinations using GPT-2
def evaluate_all_combinations(processed_test_data, learning_rates, weight_decays,
                              base_model_dir="/home/jawadkk/Brainteaser-GPT2/GPT2/"):
    for lr in learning_rates:
        for wd in weight_decays:
            model_id = f"gpt2_finetuned_lr{lr}_wd{wd}"
            model_path = os.path.join(base_model_dir, model_id)
            output_file = f"ResultsFewShot/{model_id}_results.csv"
            os.makedirs("ResultsFewShot", exist_ok=True)
            try:
                print(f"Loading model from {model_path}")
                model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
                tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                tokenizer.pad_token = tokenizer.eos_token

                accuracy = evaluate_model(model, tokenizer, processed_test_data, output_file)
                print(f"Model {model_id} Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"Error evaluating {model_id}: {e}")

# Run evaluation
evaluate_all_combinations(processed_test_data, learning_rates, weight_decays)
