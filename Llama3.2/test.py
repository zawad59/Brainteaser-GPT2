import os
import torch
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training

# Constants
CUTOFF_LEN = 512  # Increased to ensure more context is captured
MAX_NEW_TOKENS = 512
RESULTS_DIR = "llama-brainteasers-results/test"
CHECKPOINTS_DIR = "/home/jawadkk/Brainteaser-GPT2/Llama3.2/LlamaFinetuned"
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

    if 'choice_list' in item:
        ordered_choices = item['choice_list']
    else:
        distractor1 = str(item['distractor1'])
        distractor2 = str(item['distractor2'])
        distractor_unsure = str(item['distractor(unsure)'])
        choice_list = [answer, distractor1, distractor2, distractor_unsure]
        choice_order = item['choice_order']
        ordered_choices = [choice_list[i] for i in choice_order]

    system_message = (
        "You are an assistant answering riddle questions for a test. Choose the correct answer from the choices."
        " Return only the answer."
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

# Load test data
test_data = np.load('/home/jawadkk/Brainteaser-GPT2/CombinedDatasets/All_test 1.npy', allow_pickle=True).tolist()

# Main function to run predictions for all models
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

            # Prepare CSV file
            total, correct = 0, 0
            wp_total, wp_correct = 0, 0
            sp_total, sp_correct = 0, 0

            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Question ID", "Question", "Answer", "Zero-Shot Prediction", "Few-Shot Prediction"])

                # Predict for each test example
                for item in test_data:
                    question_id = item.get('id', 'N/A')
                    question = item['question']
                    answer = item['answer']

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
                    zero_shot_answer = zero_shot_prediction.split("Answer:")[-1].strip()

                    # Few-shot prediction
                    few_shot_prompt = generate_prompt(item, few_shot=True)
                    few_shot_inputs = tokenize(few_shot_prompt)
                    few_shot_inputs = {key: val.to(model.device) for key, val in few_shot_inputs.items()}
                    with torch.no_grad():
                        few_shot_outputs = model.generate(
                            **few_shot_inputs, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.2, top_p=0.9, top_k=50
                        )
                        few_shot_prediction = tokenizer.decode(few_shot_outputs[0], skip_special_tokens=True)
                    few_shot_answer = few_shot_prediction.split("Answer:")[-1].strip()

                    # Write results
                    writer.writerow([question_id, question, answer, zero_shot_answer, few_shot_answer])

                    # Accuracy calculations
                    total += 1
                    if zero_shot_answer == answer:
                        correct += 1
                        if question_id.startswith("WP"):
                            wp_total += 1
                            wp_correct += 1
                        elif question_id.startswith("SP"):
                            sp_total += 1
                            sp_correct += 1
                    elif question_id.startswith("WP"):
                        wp_total += 1
                    elif question_id.startswith("SP"):
                        sp_total += 1

            # Calculate accuracies
            overall_accuracy = (correct / total) * 100 if total > 0 else 0
            wp_accuracy = (wp_correct / wp_total) * 100 if wp_total > 0 else 0
            sp_accuracy = (sp_correct / sp_total) * 100 if sp_total > 0 else 0

            # Print accuracies
            print(f"Results for lr={lr}, wd={wd}:")
            print(f"  Overall Accuracy: {overall_accuracy:.2f}%")
            print(f"  WP Accuracy: {wp_accuracy:.2f}%")
            print(f"  SP Accuracy: {sp_accuracy:.2f}%")

            print(f"Results saved to {csv_file}")
            del model
            torch.cuda.empty_cache()


# Execute
if __name__ == "__main__":
    run_predictions()
