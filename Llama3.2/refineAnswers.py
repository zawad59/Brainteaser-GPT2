import os
import torch
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, util  # Add for cosine similarity calculations

# Constants
CUTOFF_LEN = 512
MAX_NEW_TOKENS = 50
RESULTS_DIR = "llama-brainteasers-results/test"
CHECKPOINTS_DIR = "/home/jawadkk/Brainteaser-GPT2/Llama3.2/LlamaFinetuned"
LEARNING_RATES = [0.0001]
WEIGHT_DECAYS = [0.01]

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# Load sentence embedding model for cosine similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective model for embeddings

# Function to tokenize prompt
def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors="pt"
    )


# Updated function to generate prompt
def generate_prompt(item, few_shot=True):
    question = item['question']
    choices = item['choice_list']

    system_message = (
        "You are an assistant answering riddle questions for a test. "
        "Choose the correct answer from the choices. "
        "Return the answer formatted as: [\"answer - here\"] or just the choice number (e.g., 1, 2, 3, 4)."
    )

    if few_shot:
        examples = '''
        Example 1:
        Question: Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible? 
        Choices: 1) Some daughters get married and have their own family., 2) Each daughter shares the same brother., 3) Some brothers were not loved by family., 4) None of the above.
        Answer: ["Each daughter shares the same brother."]

        Example 2:
        Question: A chess team has five players, and each player has one coach. But there are only six participants in the team. How is that possible? 
        Choices: 1) Each player shares the same coach., 2) Some players are backups., 3) Some coaches got a raise., 4) None of the above.
        Answer: ["Each player shares the same coach."]
        '''
        return (
            f"{system_message}\n\n{examples}\n"
            f"Question: {question}\nChoices: {', '.join([f'{i + 1}) {choice}' for i, choice in enumerate(choices)])}\nAnswer:"
        )

    return (
        f"{system_message}\n\n"
        f"Question: {question}\nChoices: {', '.join([f'{i + 1}) {choice}' for i, choice in enumerate(choices)])}\nAnswer:"
    )


# Updated function to refine the answer
def refine_answer(generated_answer, choices):
    # Clean up and extract formatted response
    generated_answer = generated_answer.strip()
    if "[" in generated_answer and "]" in generated_answer:
        # Extract content inside brackets
        extracted = generated_answer.split("[")[1].split("]")[0]
        refined = extracted.split(" - ")[1] if " - " in extracted else extracted
        if refined in choices:
            return refined
    elif generated_answer.isdigit():
        # Interpret as choice number
        choice_index = int(generated_answer) - 1
        if 0 <= choice_index < len(choices):
            return choices[choice_index]

    # Fall back to similarity-based refinement
    '''generated_embedding = embedding_model.encode(generated_answer, convert_to_tensor=True)
    choice_embeddings = embedding_model.encode(choices, convert_to_tensor=True)
    cosine_scores = util.cos_sim(generated_embedding, choice_embeddings)
    best_choice_idx = torch.argmax(cosine_scores).item()
    return choices[best_choice_idx]'''


# Updated logic to parse and save results
def evaluate_model(model, tokenizer, test_data, output_file, few_shot=False):
    predictions = []
    correct_predictions = 0
    total = 0

    for item in test_data:
        question_id = item['id']
        question = item['question']
        choices = item['choice_list']
        correct_answer = choices[item['label']]

        # Generate answer
        prompt = generate_prompt(item, few_shot=few_shot)
        inputs = tokenize(prompt).to(model.device)
        outputs = model.generate(
            inputs["input_ids"], max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.2, top_p=0.9, top_k=50
        )
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
        refined_answer = refine_answer(generated_answer, choices)

        is_correct = refined_answer == correct_answer
        if is_correct:
            correct_predictions += 1
        total += 1

        predictions.append({
            "Question ID": question_id,
            "Question": question,
            "Correct Answer": correct_answer,
            "Generated Answer": generated_answer,
            "Refined Answer": refined_answer,
            "Is Correct": "yes" if is_correct else "no"
        })

    accuracy = (correct_predictions / total) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    save_predictions_to_csv(predictions, output_file)
    return accuracy


# Updated writer logic for CSV
def save_predictions_to_csv(predictions, filename):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Question ID", "Question", "Correct Answer",
                                                  "Generated Answer", "Refined Answer", "Is Correct"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Results saved to {filename}")
