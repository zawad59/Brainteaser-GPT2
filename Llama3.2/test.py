import os
import torch
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Constants
CUTOFF_LEN = 512
MAX_NEW_TOKENS = 50
RESULTS_DIR = "llama-brainteasers-results/test"
CHECKPOINTS_DIR = "/home/jawadkk/Brainteaser-GPT2/Llama3.2/"
LEARNING_RATES = [0.0001]
WEIGHT_DECAYS = [0.01]

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# Function to generate zero-shot and few-shot prompts
def generate_prompt(item, few_shot=True):
    question = item['question']
    answer = item['answer']
    ordered_choices = item['choice_list']

    system_message = (
        "You are an assistant answering riddle questions for a test. "
        "Choose the correct answer from the choices below. "
        "Only return the answer as it appears in the choices."
    )
    if few_shot:
        examples = '''
        Example 1:
        Question: What is always coming but never arrives?
        Choices: ['Tomorrow', 'Yesterday', 'Now', 'Never']
        Answer: Tomorrow
        Example 2:
        Question: The more of this you take, the more you leave behind. What is it?
        Choices: ['Footsteps', 'Time', 'Money', 'Water']
        Answer: Footsteps
        '''
        return (
            f"{system_message}\n\n{examples}\n"
            f"Question: {question}\nChoices: {ordered_choices}\nAnswer:"
        )
    return (
        f"{system_message}\n\n"
        f"Question: {question}\nChoices: {ordered_choices}\nAnswer:"
    )

# Function to tokenize prompt
def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors="pt"
    )

# Function to process generated answers
def process_generated_answer(generated_answer):
    """
    Truncate the generated answer at the first full stop.
    """
    return generated_answer.split(".")[0].strip()

# Load test data
test_data = np.load('/home/jawadkk/Brainteaser-GPT2/CombinedDatasets/All_test 1.npy', allow_pickle=True).tolist()

# Main function to run predictions for all models
def run_predictions():
    for lr in LEARNING_RATES:
        for wd in WEIGHT_DECAYS:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"llama_lora_finetuned_lr{lr}_wd{wd}")
            csv_file = os.path.join(RESULTS_DIR, f"llama_lora_finetuned_results_lr{lr}_wd{wd}.csv")

            # Load model
            print(f"Loading model for lr={lr}, wd={wd}...")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                load_in_4bit=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Prepare CSV file
            total = 0
            zero_shot_correct = 0
            few_shot_correct = 0

            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Question ID", "Question Text", "Actual Answer",
                    "Generated Zero Shot Answer", "ZeroShotAns==Actual Ans(True or False)",
                    "Generated Few Shot Answer", "FewShotAns==Actual Ans(True or False)"
                ])

                # Predict for each test example
                for item in test_data:
                    question_id = item.get('id', 'N/A')
                    question = item['question']
                    answer = item['answer']
                    choices = item['choice_list']  # Get choices

                    # Zero-shot prediction
                    zero_shot_prompt = generate_prompt(item, few_shot=False)
                    zero_shot_inputs = tokenize(zero_shot_prompt)
                    zero_shot_inputs = {key: val.to(model.device) for key, val in zero_shot_inputs.items()}
                    model.eval()
                    with torch.no_grad():
                        zero_shot_outputs = model.generate(
                            **zero_shot_inputs, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.2, top_p=0.9, top_k=50
                        )
                        zero_shot_prediction = tokenizer.decode(zero_shot_outputs[0], skip_special_tokens=True)
                    zero_shot_answer = process_generated_answer(zero_shot_prediction.split("Answer:")[-1].strip())
                    zero_shot_correct = zero_shot_answer == answer

                    # Few-shot prediction
                    few_shot_prompt = generate_prompt(item, few_shot=True)
                    few_shot_inputs = tokenize(few_shot_prompt)
                    few_shot_inputs = {key: val.to(model.device) for key, val in few_shot_inputs.items()}
                    with torch.no_grad():
                        few_shot_outputs = model.generate(
                            **few_shot_inputs, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.2, top_p=0.9, top_k=50
                        )
                        few_shot_prediction = tokenizer.decode(few_shot_outputs[0], skip_special_tokens=True)
                    few_shot_answer = process_generated_answer(few_shot_prediction.split("Answer:")[-1].strip())
                    few_shot_correct = few_shot_answer == answer

                    # Update accuracy
                    total += 1
                    if zero_shot_correct:
                        zero_shot_correct += 1
                    if few_shot_correct:
                        few_shot_correct += 1

                    # Write results
                    writer.writerow([
                        question_id, question, answer,
                        zero_shot_answer, zero_shot_correct,
                        few_shot_answer, few_shot_correct
                    ])

            # Calculate accuracies
            zero_shot_accuracy = (zero_shot_correct / total) * 100 if total > 0 else 0
            few_shot_accuracy = (few_shot_correct / total) * 100 if total > 0 else 0

            # Print results
            print(f"Results for lr={lr}, wd={wd}:")
            print(f"  Zero-Shot Accuracy: {zero_shot_accuracy:.2f}%")
            print(f"  Few-Shot Accuracy: {few_shot_accuracy:.2f}%")
            print(f"Results saved to {csv_file}")
            del model
            torch.cuda.empty_cache()

# Execute
if __name__ == "__main__":
    run_predictions()
