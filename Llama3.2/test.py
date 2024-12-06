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

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# Function to generate zero-shot and few-shot prompts
def generate_prompt(item, few_shot=False):
    question = item['question']
    answer = item['answer']

    if 'choice_list' in item:
        ordered_choices = item['choice_list']  # use for preordered test dataset
    else:
        distractor1 = str(item['distractor1'])  # order the eval dataset
        distractor2 = str(item['distractor2'])
        distractor_unsure = str(item['distractor(unsure)'])
        # Create choice_list and reorder based on choice_order
        choice_list = [answer, distractor1, distractor2, distractor_unsure]
        choice_order = item['choice_order']
        ordered_choices = [choice_list[i] for i in choice_order]

    sys_msg = (
        "You are an assistant answering riddle questions for a test. For each question, "
        "you must choose an answer from the choice list. Output the chosen answer and nothing else."
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
            f"<s> [INST]{sys_msg}\n{examples}\n{question}\nChoose one of the following answers:\n"
            + "\n".join(ordered_choices) + "[/INST]</s>"
        )
    return (
        f"<s> [INST]{sys_msg}\n{question}\nChoose one of the following answers:\n"
        + "\n".join(ordered_choices) + "[/INST]</s>"
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

# Load test data
test_data = np.load('/home/jawadkk/Brainteaser-GPT2/CombinedDatasets/All_test 1.npy', allow_pickle=True).tolist()

# Main function to run predictions for all models
def run_predictions():
    for lr in LEARNING_RATES:
        for wd in WEIGHT_DECAYS:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"llama_lora_finetuned_lr{lr}_wd{wd}", "checkpoint-225")
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
                writer.writerow(["Question ID", "Question", "Answer", "Zero-Shot Prediction", "Few-Shot Prediction"])

                # Predict for each test example
                for item in test_data:
                    question_id = item.get('id', 'N/A')  # Assuming each test item has a unique ID
                    question = item['question']
                    answer = item['answer']

                    # Zero-shot prediction
                    zero_shot_prompt = generate_prompt(item, few_shot=False)
                    zero_shot_inputs = tokenize(zero_shot_prompt)
                    zero_shot_inputs = {key: val.to(model.device) for key, val in zero_shot_inputs.items()}
                    model.eval()
                    with torch.no_grad():
                        zero_shot_outputs = model.generate(**zero_shot_inputs, max_new_tokens=MAX_NEW_TOKENS)
                        zero_shot_prediction = tokenizer.decode(zero_shot_outputs[0], skip_special_tokens=True)
                    # Extract the predicted answer from the output text
                    print(f"Zero-shot Raw Prediction:\n{zero_shot_prediction}\n")
                    zero_shot_answer = zero_shot_prediction.split("[/INST]")[-1].strip()

                    # Few-shot prediction
                    few_shot_prompt = generate_prompt(item, few_shot=True)
                    few_shot_inputs = tokenize(few_shot_prompt)
                    few_shot_inputs = {key: val.to(model.device) for key, val in few_shot_inputs.items()}
                    with torch.no_grad():
                        few_shot_outputs = model.generate(**few_shot_inputs, max_new_tokens=MAX_NEW_TOKENS)
                        few_shot_prediction = tokenizer.decode(few_shot_outputs[0], skip_special_tokens=True)
                    # Extract the predicted answer from the output text
                    print(f"Few-shot Raw Prediction:\n{few_shot_prediction}\n")
                    few_shot_answer = few_shot_prediction.split("[/INST]")[-1].strip()

                    # Write results
                    writer.writerow([question_id, question, answer, zero_shot_answer, few_shot_answer])
                    print(f"Saved Results: {question_id, question, answer, zero_shot_answer, few_shot_answer}\n")

            print(f"Results saved to {csv_file}")
            del model
            torch.cuda.empty_cache()

# Execute
if __name__ == "__main__":
    run_predictions()
