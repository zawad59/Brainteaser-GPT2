import os
import torch
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training

# Constants
CUTOFF_LEN = 256
MAX_NEW_TOKENS = 50
RESULTS_DIR = "llama-brainteasers-results/FinalLlamaResultsTuned"
CHECKPOINTS_DIR = "/home/jawadkk/Brainteaser-GPT2/Llama3.2/logs_lr1e-05_wd1e-05"
LEARNING_RATES = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
WEIGHT_DECAYS = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]

# Decoding Parameters
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.2

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# Function to generate zero-shot and few-shot prompts
def generate_prompt(item, few_shot=False):
    question = item['question']
    answer = item['answer']
    choices = item.get('choice_list', [])

    sys_msg = (
        "You are an assistant answering riddle questions for a test. "
        "For each question, choose an answer from the given choices and only return the chosen answer."
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
            f"<s> [INST]{sys_msg}\n\n{examples}\n"
            f"Question: {question}\nChoices: {choices}\nAnswer:[/INST]</s>"
        )
    return (
        f"<s> [INST]{sys_msg}\n\n"
        f"Question: {question}\nChoices: {choices}\nAnswer:[/INST]</s>"
    )

# Function to tokenize prompt
def tokenize(prompt):
    return tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors="pt"
    )

# Post-process the generated answer
def refine_answer(raw_answer, choices):
    raw_answer = raw_answer.strip()
    for choice in choices:
        if choice in raw_answer:
            return choice
    return "None"  # Return "None" if no valid choice matches

# Load test data
test_data = np.load('/home/jawadkk/Brainteaser-GPT2/CombinedDatasets/All_test 1.npy', allow_pickle=True).tolist()

# Main function to run predictions for all models
def run_predictions():
    for lr in LEARNING_RATES:
        for wd in WEIGHT_DECAYS:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"llama_lora_finetuned_lr{lr}_wd{wd}")
            csv_file = os.path.join(RESULTS_DIR, f"results_lr({lr})_wd({wd}).csv")

            # Skip if results already exist
            if os.path.exists(csv_file):
                print(f"Results for lr={lr}, wd={wd} already exist. Skipping.")
                continue

            # Load model
            print(f"Loading model for lr={lr}, wd={wd}...")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                load_in_4bit=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = prepare_model_for_kbit_training(model)

            # Prepare CSV file
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Question ID", "Question", "Answer", "Choices",
                    "Zero-Shot Raw", "Zero-Shot Refined",
                    "Few-Shot Raw", "Few-Shot Refined"
                ])

                # Predict for each test example
                for item in test_data:
                    question_id = item.get('id', 'N/A')
                    question = item['question']
                    answer = item['answer']
                    choices = item['choice_list']

                    # Zero-shot prediction
                    zero_shot_prompt = generate_prompt(item, few_shot=False)
                    zero_shot_inputs = tokenize(zero_shot_prompt)
                    zero_shot_inputs = {key: val.to(model.device) for key, val in zero_shot_inputs.items()}
                    model.eval()
                    with torch.no_grad():
                        zero_shot_outputs = model.generate(
                            **zero_shot_inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            top_k=TOP_K,
                            repetition_penalty=REPETITION_PENALTY
                        )
                        zero_shot_raw = tokenizer.decode(zero_shot_outputs[0], skip_special_tokens=True)
                        zero_shot_refined = refine_answer(zero_shot_raw, choices)
                    print(f"Zero-shot Raw:\n{zero_shot_raw}\nRefined: {zero_shot_refined}\n")

                    # Few-shot prediction
                    few_shot_prompt = generate_prompt(item, few_shot=True)
                    few_shot_inputs = tokenize(few_shot_prompt)
                    few_shot_inputs = {key: val.to(model.device) for key, val in few_shot_inputs.items()}
                    with torch.no_grad():
                        few_shot_outputs = model.generate(
                            **few_shot_inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            top_k=TOP_K,
                            repetition_penalty=REPETITION_PENALTY
                        )
                        few_shot_raw = tokenizer.decode(few_shot_outputs[0], skip_special_tokens=True)
                        few_shot_refined = refine_answer(few_shot_raw, choices)
                    print(f"Few-shot Raw:\n{few_shot_raw}\nRefined: {few_shot_refined}\n")

                    # Write results
                    writer.writerow([
                        question_id, question, answer, ", ".join(choices),
                        zero_shot_raw, zero_shot_refined,
                        few_shot_raw, few_shot_refined
                    ])

            print(f"Results saved to {csv_file}")
            del model
            torch.cuda.empty_cache()

# Execute
if __name__ == "__main__":
    run_predictions()
