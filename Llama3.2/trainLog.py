import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Directory containing the TensorBoard logs (adjust this path)
LOGS_DIR = "logs_lr1e-05_wd1e-05/llama_lora_finetuned_lr0.0001_wd0.0001/runs"
OUTPUT_CSV = "training_metrics.csv"

# Tags to extract (adjust based on your logging setup)
TRAIN_LOSS_TAG = "loss"
EVAL_LOSS_TAG = "eval_loss"
model_id = "llama_lora_finetuned_lr0.0001_wd0.0001"

def extract_metrics_from_events(logs_dir, output_csv):
    # Prepare the CSV file
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model_ID", "Step", "Train Loss", "Eval Loss"])  # Header

        # Walk through the logs directory
        for root, dirs, files in os.walk(logs_dir):
            for file_name in files:
                if "events.out.tfevents" in file_name:
                    log_path = os.path.join(root, file_name)
                    print(f"Processing log file: {log_path}")

                    # Load TensorBoard log file
                    event_acc = EventAccumulator(log_path)
                    event_acc.Reload()

                    # Extract metrics
                    train_loss_data = event_acc.Scalars(TRAIN_LOSS_TAG) if TRAIN_LOSS_TAG in event_acc.Tags()["scalars"] else []
                    eval_loss_data = event_acc.Scalars(EVAL_LOSS_TAG) if EVAL_LOSS_TAG in event_acc.Tags()["scalars"] else []

                    # Combine steps from both training and evaluation
                    steps = set([entry.step for entry in train_loss_data] + [entry.step for entry in eval_loss_data])

                    # Write metrics to CSV
                    for step in sorted(steps):
                        train_loss = next((entry.value for entry in train_loss_data if entry.step == step), None)
                        eval_loss = next((entry.value for entry in eval_loss_data if entry.step == step), None)
                        writer.writerow([model_id, step, train_loss, eval_loss])

    print(f"Metrics extracted and saved to: {output_csv}")

# Execute the script
if __name__ == "__main__":
    extract_metrics_from_events(LOGS_DIR, OUTPUT_CSV)
