import os
import numpy as np
import torch
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util  # For cosine similarity

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

# Few-shot examples
FEW_SHOT_EXAMPLES = """
Example 1:
Q: Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?
Choices: ['Some daughters get married and have their own family.', 'Each daughter shares the same brother.', 'Some brothers were not loved by family and moved away.', 'None of above.']
A: Each daughter shares the same brother.

Example 2:
Q: The six daughters of Mr. and Mrs. Mustard each have one brother. However, the family only consists of nine people; how is that possible?
Choices: ['Some brothers were not loved by family and moved away.', 'Some daughters get married and have their own family.', 'Each daughter shares the same brother.', 'None of above.']
A: Each daughter shares the same brother.
"""

# Generate answers using the LLaMA model
def generate_answer(model, tokenizer, question, choices, few_shot=False):
    if few_shot:
        prompt = f"{FEW_SHOT_EXAMPLES}\nQ: {question}\nChoices: {', '.join(choices)}\nA:"
    else:
        prompt = f"Q: {question}\nChoices: {', '.join(choices)}\nA:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.5,
        temperature=0.0,
        num_beams=1,
        do_sample=False
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_part = generated_text.split("A:")[-1].strip()
    return answer_part.split("\n")[0].strip()

# Refine prediction using cosine similarity
def refine_prediction_with_similarity(generated_answer, choices):
    if generated_answer in choices:
        return generated_answer
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    best_index = torch.argmax(cosine_similarities).item()
    return choices[best_index]

# Evaluate the LLaMA model
def evaluate_model(model, tokenizer, test_data, output_file, few_shot=False):
    predictions = []
    correct_predictions = 0
    for item in test_data:
        question_id = item['id']
        question = item['text']
        choices = item['choices']
        correct_answer = item['correct_answer']

        generated_answer = generate_answer(model, tokenizer, question, choices, few_shot=few_shot)
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
    print(f"{'Few-Shot' if few_shot else 'Zero-Shot'} Test Accuracy: {accuracy:.4f}")
    save_predictions_to_csv(predictions, output_file)
    return accuracy

# Save predictions to CSV
def save_predictions_to_csv(predictions, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question_ID", "Question_Text", "Choices",
                                                  "Generated Answer", "Refined Answer",
                                                  "Correct Answer", "Predicted == Correct"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Predictions saved to {filename}")

# Evaluate all combinations using LLaMA
def evaluate_all_combinations(processed_test_data, learning_rates, weight_decays,
                              base_model_dir="/home/jawadkk/Brainteaser-GPT2/Llama3.2/"):
    for lr in learning_rates:
        for wd in weight_decays:
            model_id = f"llama_finetuned_lr{lr}_wd{wd}"
            model_path = os.path.join(base_model_dir, model_id)
            zero_shot_output_file = f"ResultsFewShot/{model_id}_zero_shot_results.csv"
            few_shot_output_file = f"ResultsFewShot/{model_id}_few_shot_results.csv"
            os.makedirs("ResultsFewShot", exist_ok=True)
            try:
                print(f"Loading model from {model_path}")
                model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                tokenizer.pad_token = tokenizer.eos_token

                zero_shot_accuracy = evaluate_model(model, tokenizer, processed_test_data, zero_shot_output_file, few_shot=False)
                few_shot_accuracy = evaluate_model(model, tokenizer, processed_test_data, few_shot_output_file, few_shot=True)
                print(f"Model {model_id} Zero-Shot Accuracy: {zero_shot_accuracy:.4f}")
                print(f"Model {model_id} Few-Shot Accuracy: {few_shot_accuracy:.4f}")
            except Exception as e:
                print(f"Error evaluating {model_id}: {e}")

# Run evaluation
evaluate_all_combinations(processed_test_data, [0.01], [0.0001])
