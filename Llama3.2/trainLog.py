import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Directories and files
LOGS_DIR = "logs_lr1e-05_wd1e-05/llama_lora_finetuned_lr0.0001_wd0.0001/runs"
OUTPUT_DIR = "/home/jawadkk/Brainteaser-GPT2/Llama3.2/TrainLogs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "training_metrics.csv")

# Tags for TensorBoard metrics
TRAIN_LOSS_TAG = "train/loss"  # Adjusted based on inspected tags
EVAL_LOSS_TAG = "eval/loss"
MODEL_ID = "llama_lora_finetuned_lr0.0001_wd0.0001"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def inspect_tags(logs_dir):
    """
    Inspect available tags in the TensorBoard logs.
    """
    print("Inspecting available tags in TensorBoard logs...")
    for root, dirs, files in os.walk(logs_dir):
        for file_name in files:
            if "events.out.tfevents" in file_name:
                log_path = os.path.join(root, file_name)
                print(f"Inspecting log file: {log_path}")
                event_acc = EventAccumulator(log_path)
                event_acc.Reload()
                print(f"Available tags in {file_name}: {event_acc.Tags()['scalars']}")

def extract_metrics_from_events(logs_dir, output_csv):
    """
    Extract training and evaluation metrics from TensorBoard event logs and save to CSV.
    """
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model_ID", "Step", "Train Loss", "Eval Loss"])  # CSV header

        for root, dirs, files in os.walk(logs_dir):
            for file_name in files:
                if "events.out.tfevents" in file_name:
                    log_path = os.path.join(root, file_name)
                    print(f"Processing log file: {log_path}")

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

                    # Debugging: Check if data is being loaded
                    if not train_loss_data and not eval_loss_data:
                        print(f"No data found in {file_name} for tags: {TRAIN_LOSS_TAG}, {EVAL_LOSS_TAG}")
                        continue

                    # Combine steps from both training and evaluation
                    steps = set(
                        [entry.step for entry in train_loss_data]
                        + [entry.step for entry in eval_loss_data]
                    )

                    # Write data to CSV
                    for step in sorted(steps):
                        train_loss = next(
                            (entry.value for entry in train_loss_data if entry.step == step),
                            None,
                        )
                        eval_loss = next(
                            (entry.value for entry in eval_loss_data if entry.step == step),
                            None,
                        )
                        writer.writerow([MODEL_ID, step, train_loss, eval_loss])
                        print(f"Step: {step}, Train Loss: {train_loss}, Eval Loss: {eval_loss}")

    print(f"Metrics extracted and saved to: {output_csv}")


if __name__ == "__main__":
    # Inspect available tags
    inspect_tags(LOGS_DIR)

    # Extract metrics
    extract_metrics_from_events(LOGS_DIR, OUTPUT_CSV)
