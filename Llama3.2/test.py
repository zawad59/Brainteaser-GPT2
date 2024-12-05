import os
import torch
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, util  # For cosine similarity

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
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate zero-shot and few-shot prompts
def generate_prompt(item, few_shot=False):
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

    sys_msg = (
        "You are an assistant answering riddle questions for a test. "
        "Choose the correct answer from the choices provided. "
        "Output the answer exactly as it appears in the choices."
    )
    
    examples = ""
    if few_shot:
        examples = '''
        Example 1:
        Question: Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?
        Choices: ['Each daughter shares the same brother.', 'Some daughters get married.', 'Some brothers were not loved by family.', 'None of above.']
        Answer: Each daughter shares the same brother.
        
        Example 2:
        Question: What TV program should you watch in the bathtub?
        Choices: ['Soap operas.', 'Sports live.', 'Talk show.', 'None of above.']
        Answer: Soap operas.
        
        ---
        '''
    return (
        f"{sys_msg}\n\n{examples}"
        f"Question: {question}\n"
        f"Choices: {', '.join(ordered_choices)}\n"
        "Answer:"
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

# Function to refine answer using cosine similarity and enforce valid output
def refine_answer(generated_answer, choices):
    # Clean up generated answer
    generated_answer = generated_answer.strip()
    generated_answer = generated_answer.replace("[INST]", "").replace("</s>", "").strip()
    if "Answer:" in generated_answer:
        generated_answer = generated_answer.split("Answer:")[-1].strip()
    # Validate against choices
    if generated_answer in choices:
        return generated_answer
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
                    zero_shot_answer = refine_answer(zero_shot_prediction, choices)
                    refined_zero_shot_correct = zero_shot_answer == answer

                    # Few-shot prediction
                    few_shot_prompt = generate_prompt(item, few_shot=True)
                    few_shot_inputs = tokenize(few_shot_prompt)
                    few_shot_inputs = {key: val.to(model.device) for key, val in few_shot_inputs.items()}
                    with torch.no_grad():
                        few_shot_outputs = model.generate(
                            **few_shot_inputs, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.2, top_p=0.9, top_k=50
                        )
                        few_shot_prediction = tokenizer.decode(few_shot_outputs[0], skip_special_tokens=True)
                    few_shot_answer = refine_answer(few_shot_prediction, choices)
                    refined_few_shot_correct = few_shot_answer == answer

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
                        zero_shot_prediction, zero_shot_answer, refined_zero_shot_correct,
                        few_shot_prediction, few_shot_answer, refined_few_shot_correct
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
