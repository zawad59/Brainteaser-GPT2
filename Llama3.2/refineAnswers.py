import os
import torch
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training

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

# Function to generate prompts
def generate_prompt(item, few_shot=True):
    question = item['question']
    choices = item['choice_list']
    system_message = (
        "You are an assistant answering riddle questions for a test. Choose the correct answer from the choices. "
        "Return only the choice number or the exact answer in the format: Answer: (your choice)."
    )
    if few_shot:
        examples = '''
        Example 1:
        Question: Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible? 
        Choices: ['Each daughter shares the same brother.', 'Some daughters get married.', 'Some brothers were not loved by family.', 'None of the above.']
        Answer: Each daughter shares the same brother.
        
        Example 2:
        Question: A chess team has five players, and each player has one coach. But there are only six participants in the team. How is that possible? 
        Choices: ['Each player shares the same coach.', 'Some players are backups.', 'Some coaches got a raise.', 'None of the above.']
        Answer: Each player shares the same coach.
        '''
        return f"{system_message}\n\n{examples}\nQuestion: {question}\nChoices: {', '.join(choices)}\nAnswer:"
    return f"{system_message}\nQuestion: {question}\nChoices: {', '.join(choices)}\nAnswer:"

# Tokenizer wrapper
def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors="pt"
    )

# Function to clean and validate generated answers
def refine_answer(generated_answer, choices):
    # Clean up and extract formatted response
    generated_answer = generated_answer.strip()
    if "Answer:" in generated_answer:
        generated_answer = generated_answer.split("Answer:")[-1].strip()
    if generated_answer in choices:
        return generated_answer
    elif generated_answer.isdigit():
        idx = int(generated_answer) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    return generated_answer  # Return as-is if no match

# Load test data
test_data = np.load('/home/jawadkk/Brainteaser-GPT2/CombinedDatasets/All_test 1.npy', allow_pickle=True).tolist()

# Main function to run predictions
def run_predictions():
    for lr in LEARNING_RATES:
        for wd in WEIGHT_DECAYS:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR)
            csv_file = os.path.join(RESULTS_DIR, f"llama_lora_finetuned_results_lr{lr}_wd{wd}.csv")

            # Load model
            print(f"Loading model for lr={lr}, wd={wd}...")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                load_in_4bit=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = prepare_model_for_kbit_training(model)

            # Initialize counters
            total = 0
            zero_shot_correct = 0
            few_shot_correct = 0

            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Question ID", "Question", "Correct Answer", "Choices",
                    "Generated Zero-Shot", "Refined Zero-Shot", "Zero-Shot Correct",
                    "Generated Few-Shot", "Refined Few-Shot", "Few-Shot Correct"
                ])

                # Predict for each test example
                for item in test_data:
                    question_id = item.get('id', 'N/A')
                    question = item['question']
                    correct_answer = item['answer']
                    choices = item['choice_list']

                    # Zero-shot prediction
                    zero_shot_prompt = generate_prompt(item, few_shot=False)
                    zero_shot_inputs = tokenize(zero_shot_prompt)
                    zero_shot_inputs = {key: val.to(model.device) for key, val in zero_shot_inputs.items()}
                    model.eval()
                    with torch.no_grad():
                        zero_shot_outputs = model.generate(
                            **zero_shot_inputs, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.2, top_p=0.9, top_k=50
                        )
                        zero_shot_generated = tokenizer.decode(zero_shot_outputs[0], skip_special_tokens=True)
                    zero_shot_refined = refine_answer(zero_shot_generated, choices)
                    zero_shot_is_correct = zero_shot_refined == correct_answer

                    # Few-shot prediction
                    few_shot_prompt = generate_prompt(item, few_shot=True)
                    few_shot_inputs = tokenize(few_shot_prompt)
                    few_shot_inputs = {key: val.to(model.device) for key, val in few_shot_inputs.items()}
                    with torch.no_grad():
                        few_shot_outputs = model.generate(
                            **few_shot_inputs, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.2, top_p=0.9, top_k=50
                        )
                        few_shot_generated = tokenizer.decode(few_shot_outputs[0], skip_special_tokens=True)
                    few_shot_refined = refine_answer(few_shot_generated, choices)
                    few_shot_is_correct = few_shot_refined == correct_answer

                    # Update counters
                    total += 1
                    if zero_shot_is_correct:
                        zero_shot_correct += 1
                    if few_shot_is_correct:
                        few_shot_correct += 1

                    # Write results
                    writer.writerow([
                        question_id, question, correct_answer, ", ".join(choices),
                        zero_shot_generated, zero_shot_refined, zero_shot_is_correct,
                        few_shot_generated, few_shot_refined, few_shot_is_correct
                    ])

            # Calculate and print accuracies
            zero_shot_accuracy = (zero_shot_correct / total) * 100 if total > 0 else 0
            few_shot_accuracy = (few_shot_correct / total) * 100 if total > 0 else 0
            print(f"Results for lr={lr}, wd={wd}:")
            print(f"  Zero-Shot Accuracy: {zero_shot_accuracy:.2f}%")
            print(f"  Few-Shot Accuracy: {few_shot_accuracy:.2f}%")
            print(f"Results saved to {csv_file}")
            del model
            torch.cuda.empty_cache()

# Execute
if __name__ == "__main__":
    run_predictions()
