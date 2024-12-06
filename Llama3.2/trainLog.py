import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Define the directories and output CSV path
LOGS_DIR = "logs_lr1e-05_wd1e-05/llama_lora_finetuned_lr0.0001_wd0.0001/runs"  # Update to your TensorBoard log directory
OUTPUT_CSV = "training_metrics.csv"
TRAIN_LOSS_TAG = "loss"
EVAL_LOSS_TAG = "eval/loss"
MODEL_ID = "llama_lora_finetuned_lr0.0001_wd0.0001"

# Function to extract metrics from TensorBoard logs
def extract_metrics_from_events(logs_dir, output_csv):
    # Ensure the parent directory for the CSV exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Open the CSV file for writing
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model_ID", "Step", "Train Loss", "Eval Loss"])  # Write the header

        # Walk through the logs directory to find TensorBoard event files
        for root, dirs, files in os.walk(logs_dir):
            for file_name in files:
                if "events.out.tfevents" in file_name:
                    log_path = os.path.join(root, file_name)
                    print(f"Processing log file: {log_path}")

                    # Load the TensorBoard log
                    event_acc = EventAccumulator(log_path)
                    event_acc.Reload()

                    # Extract data for train loss and eval loss
                    train_loss_data = event_acc.Scalars(TRAIN_LOSS_TAG) if TRAIN_LOSS_TAG in event_acc.Tags()["scalars"] else []
                    eval_loss_data = event_acc.Scalars(EVAL_LOSS_TAG) if EVAL_LOSS_TAG in event_acc.Tags()["scalars"] else []

                    # Combine steps from both metrics
                    all_steps = set(entry.step for entry in train_loss_data) | set(entry.step for entry in eval_loss_data)

                    # Write each step's data to the CSV
                    for step in sorted(all_steps):
                        train_loss = next((entry.value for entry in train_loss_data if entry.step == step), None)
                        eval_loss = next((entry.value for entry in eval_loss_data if entry.step == step), None)
                        writer.writerow([MODEL_ID, step, train_loss, eval_loss])

    print(f"Metrics extracted and saved to: {output_csv}")

# Execute the script
if __name__ == "__main__":
    extract_metrics_from_events(LOGS_DIR, OUTPUT_CSV)
