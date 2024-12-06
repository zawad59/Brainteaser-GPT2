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
RESULTS_DIR = "llama-brainteasers-results/FinalLlamaResultsTuned"
CHECKPOINTS_DIR = "/home/jawadkk/Brainteaser-GPT2/Llama3.2/logs_lr1e-05_wd1e-05/"
LEARNING_RATES = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
WEIGHT_DECAYS = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]

# Best hyperparameters based on evaluation
BEST_TOP_P = 0.9
BEST_TOP_K = 50
BEST_TEMPERATURE = 0.9
BEST_REPETITION_PENALTY = 1.2

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# Load sentence embedding model for cosine similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate zero-shot and few-shot prompts
def generate_prompt(item, few_shot=True):
    question = item['question']
    choices = item['choice_list']
    system_message = (
        "You are a highly accurate assistant for answering riddles. "
        "Your task is to choose the correct answer from the given choices. "
        "Please only return the answer exactly as it appears in the choices."
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
            f"Question: {question}\nChoices: {choices}\nAnswer:"
        )
    return f"{system_message}\n\nQuestion: {question}\nChoices: {choices}\nAnswer:"

# Function to tokenize prompt
def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors="pt"
    )

# Function to refine answer using post-processing and cosine similarity
def refine_answer(generated_answer, choices):
    # Clean up generated answer
    artifacts = ["istr", "actor", "'", ",", ":", " by", " to", " the", ".", "answer's"]
    for artifact in artifacts:
        generated_answer = generated_answer.replace(artifact, "").strip()

    # Validate against choices
    if generated_answer in choices:
        return generated_answer  # Valid answer

    # Use cosine similarity to find the closest valid choice
    '''generated_embedding = embedding_model.encode(generated_answer, convert_to_tensor=True)
    choice_embeddings = embedding_model.encode(choices, convert_to_tensor=True)
    similarity_scores = util.cos_sim(generated_embedding, choice_embeddings)
    best_choice_idx = similarity_scores.argmax().item()
    return choices[best_choice_idx]'''

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
            model = prepare_model_for_kbit_training(model)

            # Prepare CSV file
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Question ID", "Question", "Answer", "Choices",
                    "Generated Zero-Shot", "Refined Zero-Shot", "Zero-Shot Correct",
                    "Generated Few-Shot", "Refined Few-Shot", "Few-Shot Correct"
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
                            top_p=BEST_TOP_P,
                            top_k=BEST_TOP_K,
                            temperature=BEST_TEMPERATURE,
                            repetition_penalty=BEST_REPETITION_PENALTY
                        )
                        zero_shot_prediction = tokenizer.decode(zero_shot_outputs[0], skip_special_tokens=True)
                    refined_zero_shot_answer = refine_answer(zero_shot_prediction.split("Answer:")[-1].strip(), choices)
                    print(f"Answer: {answer}\nGenerated Zero-Shot Answer: {refined_zero_shot_answer}\n")


                    zero_shot_correct = refined_zero_shot_answer == answer

                    # Few-shot prediction
                    few_shot_prompt = generate_prompt(item, few_shot=True)
                    few_shot_inputs = tokenize(few_shot_prompt)
                    few_shot_inputs = {key: val.to(model.device) for key, val in few_shot_inputs.items()}
                    with torch.no_grad():
                        few_shot_outputs = model.generate(
                            **few_shot_inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            top_p=BEST_TOP_P,
                            top_k=BEST_TOP_K,
                            temperature=BEST_TEMPERATURE,
                            repetition_penalty=BEST_REPETITION_PENALTY
                        )
                        few_shot_prediction = tokenizer.decode(few_shot_outputs[0], skip_special_tokens=True)
                    refined_few_shot_answer = refine_answer(few_shot_prediction.split("Answer:")[-1].strip(), choices)
                    print(f"Answer: {answer}\nGenerated Few-Shot Answer: {refined_few_shot_answer}\n")
                    few_shot_correct = refined_few_shot_answer == answer

                    # Write results
                    writer.writerow([
                        question_id, question, answer, ", ".join(choices),
                        zero_shot_prediction, refined_zero_shot_answer, zero_shot_correct,
                        few_shot_prediction, refined_few_shot_answer, few_shot_correct
                    ])

            print(f"Results saved to {csv_file}")
            del model
            torch.cuda.empty_cache()

# Execute
if __name__ == "__main__":
    run_predictions()
