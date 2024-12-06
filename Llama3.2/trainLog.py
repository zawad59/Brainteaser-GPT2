import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Base directories and output CSV file
BASE_DIR = "/home/jawadkk/Brainteaser-GPT2/Llama3.2"
OUTPUT_CSV = os.path.join(BASE_DIR, "all_training_metrics.csv")

# Hyperparameters
LEARNING_RATES = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
WEIGHT_DECAYS = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]

# Tags for TensorBoard metrics
TRAIN_LOSS_TAG = "train/loss"  # Adjusted based on inspected tags
EVAL_LOSS_TAG = "eval/loss"

def extract_metrics_from_events(log_dir, model_id, writer):
    """
    Extract metrics for a single model from TensorBoard event logs.
    """
    for root, dirs, files in os.walk(log_dir):
        for file_name in files:
            if "events.out.tfevents" in file_name:
                log_path = os.path.join(root, file_name)
                print(f"Processing log file: {log_path}")

                # Load TensorBoard log
                event_acc = EventAccumulator(log_path)
                event_acc.Reload()

                # Extract metrics
                train_loss_data = (
                    event_acc.Scalars(TRAIN_LOSS_TAG)
                    if TRAIN_LOSS_TAG in event_acc.Tags()["scalars"]
                    else []
                )
                eval_loss_data = (
                    event_acc.Scalars(EVAL_LOSS_TAG)
                    if EVAL_LOSS_TAG in event_acc.Tags()["scalars"]
                    else []
                )

                # Combine steps from both training and evaluation
                steps = set(
                    [entry.step for entry in train_loss_data]
                    + [entry.step for entry in eval_loss_data]
                )

                # Write metrics to CSV
                for step in sorted(steps):
                    train_loss = next(
                        (entry.value for entry in train_loss_data if entry.step == step),
                        None,
                    )
                    eval_loss = next(
                        (entry.value for entry in eval_loss_data if entry.step == step),
                        None,
                    )
                    writer.writerow([model_id, step, train_loss, eval_loss])
                return  # Stop after processing the first log file

def process_all_folders(base_dir, output_csv):
    """
    Process all folders for different learning rate and weight decay combinations.
    """
    # Prepare CSV file
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model_ID", "Step", "Train Loss", "Eval Loss"])  # Header

        # Iterate through all combinations
        for lr in LEARNING_RATES:
            for wd in WEIGHT_DECAYS:
                model_id = f"llama_lora_finetuned_lr{lr}_wd{wd}"
                log_dir = os.path.join(base_dir, model_id)

                if os.path.exists(log_dir):
                    print(f"Processing folder: {model_id}")
                    extract_metrics_from_events(log_dir, model_id, writer)
                else:
                    print(f"Folder not found: {model_id}")

    print(f"All metrics extracted and saved to: {output_csv}")

if __name__ == "__main__":
    process_all_folders(BASE_DIR, OUTPUT_CSV)
