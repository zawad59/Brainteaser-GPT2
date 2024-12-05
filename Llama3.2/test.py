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
CHECKPOINTS_DIR = "/home/jawadkk/Brainteaser-GPT2/Llama3.2/"
LEARNING_RATES = [0.01]
WEIGHT_DECAYS = [0.0001]

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# Load sentence embedding model for cosine similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective model for embeddings

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
    "You are an assistant for a test. Your task is to answer riddle questions by selecting a correct answer from the given choices."
    "Only output the chosen answer without repeating instructions."
)
    if few_shot:
        examples = '''
                    Example 1:
                    SP-0 
                    Question: Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible? 
                    Choices:  ['Some daughters get married and have their own family.', 'Each daughter shares the same brother.', 'Some brothers were not loved by family and moved away.', 'None of above.'] 
                    Answer: Each daughter shares the same brother. 

                    SP-0_SR 
                    Question: The six daughters of Mr. and Mrs. Mustard each have one brother. However, the family only consists of nine people; how is that possible? 
                    Choices: ['Some brothers were not loved by family and moved away.', 'Some daughters get married and have their own family.', 'Each daughter shares the same brother.', 'None of above.'] 
                    Answer: Each daughter shares the same brother. 

                    SP-0_CR 
                    Question: A chess team has five players, and each player has one coach. But there are only six participants in the team. How is that possible? 
                    Choices: ['Each player shares the same coach.', 'Some players are backups and not allowed to play.', 'Some coaches get a raise.', 'None of above.'] 
                    Answer: Each player shares the same coach. 

                    Example 2:
                    WP-115 
                    Question: What TV program should you watch in the bathtub?
                    Choices: ['Soap operas.', 'Sports live.', 'Talk show.', 'None of above.']
                    Answer: Soap operas. 
                    
                    WP-115_SR 
                    Question: What TV show should you watch in the tub?, 
                    Choices: ['Soap operas.', 'Talk show.', 'Sports live.', 'None of above.']
                    Answer: Soap operas.
                    
                    WP-115_CR
                    Question: What TV show should people in serious denial watch ?,  
                    Choices: ['Reality TV shows', 'Sports live.', 'Soap operas.', 'None of above.'],
                    Answer: Reality TV shows.
                    
                    '''
        return (
            f"<s> [INST]{sys_msg}\n{examples}{question}\nChoose one of the following answers from the following choices:\n"
            + "\n".join(ordered_choices) + "[/INST]</s>" + answer + "</s>"
        )
    return (
        f"<s> [INST]{sys_msg}\n{question}\nChoose one of the following answers from the following choices:\n"
        + "\n".join(ordered_choices) + "[/INST]</s>" + answer + "</s>"
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

# Function to refine answer using cosine similarity and enforce valid output
def refine_answer(generated_answer, choices):
    # Clean up generated answer
    generated_answer = generated_answer.strip()  # Remove extra spaces
    generated_answer = generated_answer.replace("Question:", "").replace("Answer:", "").strip()  # Remove prefixes
    # Validate against choices
    if generated_answer in choices:
        return generated_answer  # Valid answer
    # Otherwise, refine using cosine similarity
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
            total = 0
            zero_shot_correct = 0
            few_shot_correct = 0
            combined_correct = 0

            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Question ID", "Question", "Answer", "Choices",
                    "Generated Zero-Shot", "Refined Zero-Shot", "Refined Zero-Shot Correct",
                    "Generated Few-Shot", "Refined Few-Shot", "Refined Few-Shot Correct"
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
                    zero_shot_answer = zero_shot_prediction.split("Answer:")[-1].strip()

                    # Refine zero-shot prediction (ensure it's one of the choices)
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

                    # Refine few-shot prediction (ensure it's one of the choices)
                    refined_few_shot_answer = refine_answer(few_shot_answer, choices)
                    refined_few_shot_correct = refined_few_shot_answer == answer

                    # Update accuracy
                    total += 1
                    if refined_zero_shot_correct:
                        zero_shot_correct += 1
                    if refined_few_shot_correct:
                        few_shot_correct += 1
                    if refined_zero_shot_correct or refined_few_shot_correct:
                        combined_correct += 1

                    # Write results
                    writer.writerow([
                        question_id, question, answer, ", ".join(choices),
                        zero_shot_answer, refined_zero_shot_answer, refined_zero_shot_correct,
                        few_shot_answer, refined_few_shot_answer, refined_few_shot_correct
                    ])

            # Calculate accuracies
            zero_shot_accuracy = (zero_shot_correct / total) * 100 if total > 0 else 0
            few_shot_accuracy = (few_shot_correct / total) * 100 if total > 0 else 0
            combined_accuracy = (combined_correct / total) * 100 if total > 0 else 0

            # Print results
            print(f"Results for lr={lr}, wd={wd}:")
            print(f"  Refined Zero-Shot Accuracy: {zero_shot_accuracy:.2f}%")
            print(f"  Refined Few-Shot Accuracy: {few_shot_accuracy:.2f}%")
            print(f"  Combined Accuracy (Zero-Shot or Few-Shot Correct): {combined_accuracy:.2f}%")
            print(f"Results saved to {csv_file}")
            del model
            torch.cuda.empty_cache()

# Execute
if __name__ == "__main__":
    run_predictions()
