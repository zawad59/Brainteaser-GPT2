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
        " Return only the answer. Don't generate anything which is not in the answer choices or in other words don't generate something random"
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

# Function to refine answer using cosine similarity
def refine_answer(generated_answer, choices):
    generated_embedding = embedding_model.encode(generated_answer, convert_to_tensor=True)
    choice_embeddings = embedding_model.encode(choices, convert_to_tensor=True)
    cosine_scores = util.cos_sim(generated_embedding, choice_embeddings)
    best_choice_idx = torch.argmax(cosine_scores).item()
    return choices[best_choice_idx]

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
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Question ID", "Question", "Answer", 
                    "Refined Zero-Shot", "Refined Zero-Shot Correct",
                    "Refined Few-Shot", "Refined Few-Shot Correct"
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
                            **zero_shot_inputs, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.2, top_p=0.9, top_k=50
                        )
                        zero_shot_prediction = tokenizer.decode(zero_shot_outputs[0], skip_special_tokens=True)
                    zero_shot_answer = zero_shot_prediction.split("Answer:")[-1].strip()

                    # Refine zero-shot prediction
                    refined_zero_shot_answer = refine_answer(zero_shot_answer, choices)
                    refined_zero_shot_correct = refined_zero_shot_answer == answer

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

                    # Refine few-shot prediction
                    refined_few_shot_answer = refine_answer(few_shot_answer, choices)
                    refined_few_shot_correct = refined_few_shot_answer == answer

                    # Write results
                    writer.writerow([
                        question_id, question, answer, 
                        refined_zero_shot_answer, refined_zero_shot_correct,
                        refined_few_shot_answer, refined_few_shot_correct
                    ])

            print(f"Results saved to {csv_file}")
            del model
            torch.cuda.empty_cache()

# Execute
if __name__ == "__main__":
    run_predictions()
