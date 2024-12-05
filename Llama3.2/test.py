import os
import numpy as np
import torch
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# Constants
CUTOFF_LEN = 512 
MAX_NEW_TOKENS = 50
RESULTS_DIR = "llama-brainteasers-results/test"
CHECKPOINTS_DIR = "/home/jawadkk/Brainteaser-GPT2/Llama3.2/"
LEARNING_RATES = [0.01]
WEIGHT_DECAYS = [0.0001]

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token
embedder = SentenceTransformer('all-MiniLM-L6-v2').to("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
test_data = np.load("/home/jawadkk/Brainteaser-GPT2/CombinedDatasets/All_test 1.npy", allow_pickle=True).tolist()

# Generate few-shot examples
FEW_SHOT_EXAMPLES = """
Example 1:
Q: Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?
Choices: ['Some daughters get married and have their own family.', 'Each daughter shares the same brother.', 'Some brothers were not loved by family.', 'None of above.']
A: Each daughter shares the same brother.

Example 2:
Q: The six daughters of Mr. and Mrs. Mustard each have one brother. However, the family only consists of nine people; how is that possible?
Choices: ['Some brothers were not loved by family and moved away.', 'Some daughters get married and have their own family.', 'Each daughter shares the same brother.', 'None of above.']
A: Each daughter shares the same brother.
"""

# Generate prompt
def generate_prompt(item, few_shot=False):
    question = item['question']
    choices = item['choice_list']
    if few_shot:
        return f"{FEW_SHOT_EXAMPLES}\nQ: {question}\nChoices: {', '.join(choices)}\nA:"
    return f"Q: {question}\nChoices: {', '.join(choices)}\nA:"

# Tokenize prompt
def tokenize(prompt):
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CUTOFF_LEN, padding=True)

# Refine answer using cosine similarity
def refine_answer(generated_answer, choices):
    if generated_answer in choices:
        return generated_answer
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    return choices[torch.argmax(cosine_similarities).item()]

# Evaluate model
def evaluate_model(model, tokenizer, test_data, output_file, few_shot=False):
    predictions = []
    for item in test_data:
        question_id = item['id']
        question = item['question']
        choices = item['choice_list']
        correct_answer = choices[item['label']]

        # Generate answer
        prompt = generate_prompt(item, few_shot=few_shot)
        inputs = tokenize(prompt).to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(
            inputs["input_ids"], max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.2, top_p=0.9, top_k=50
        )
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("A:")[-1].strip()
        refined_answer = refine_answer(generated_answer, choices)

        predictions.append([
            question_id, question, correct_answer,
            generated_answer, refined_answer
        ])

    # Write results to CSV
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Question_ID", "Question", "Correct Answer", "Generated Answer", "Refined Answer"])
        writer.writerows(predictions)
    print(f"Results saved to {output_file}")

# Evaluate all combinations
def evaluate_all_combinations(test_data, learning_rates, weight_decays, base_model_dir=CHECKPOINTS_DIR):
    for lr in learning_rates:
        for wd in weight_decays:
            model_dir = os.path.join(base_model_dir, f"llama_lora_finetuned_lr{lr}_wd{wd}")
            zero_shot_output_file = f"{RESULTS_DIR}/llama_zero_shot_lr{lr}_wd{wd}.csv"
            few_shot_output_file = f"{RESULTS_DIR}/llama_few_shot_lr{lr}_wd{wd}.csv"

            try:
                model = AutoModelForCausalLM.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
                evaluate_model(model, tokenizer, test_data, zero_shot_output_file, few_shot=False)
                evaluate_model(model, tokenizer, test_data, few_shot_output_file, few_shot=True)
            except Exception as e:
                print(f"Error evaluating model at {model_dir}: {e}")

# Run evaluation
evaluate_all_combinations(test_data, LEARNING_RATES, WEIGHT_DECAYS)
