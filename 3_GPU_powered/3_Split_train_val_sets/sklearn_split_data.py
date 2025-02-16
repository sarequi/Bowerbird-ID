import os
import shutil
import random
from sklearn.model_selection import train_test_split
import pandas as pd  

source_dir = "/gpfs/data/fs72607/juarezs98/masked_frames/"
train_dir  = "/gpfs/data/fs72607/juarezs98/train_val_test_data/Training"
val_dir    = "/gpfs/data/fs72607/juarezs98/train_val_test_data/Validation"
test_dir   = "/gpfs/data/fs72607/juarezs98/train_val_test_data/Testing"
log_file   = "/gpfs/data/fs72607/juarezs98/train_val_test_data/processed_bird_ids.log"

metadata_file = "/gpfs/data/fs72607/juarezs98/masked_frames/masked_frames_metadata.csv"               
new_metadata_file  = "/gpfs/data/fs72607/juarezs98/train_val_test_data/metadata_split_train_val_test.csv"  # New metadata output path

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read already processed bird IDs from log
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        processed_bird_ids = set(line.strip() for line in f)
else:
    processed_bird_ids = set()

summary = {}
# Mapping from (bird_id, frame_name) to dataset assignment
frame_dataset_mapping = {}

# Iterate through each bird folder
for bird_id in os.listdir(source_dir):
    if bird_id in processed_bird_ids:
        print(f"Skipping already processed bird '{bird_id}'.")
        continue

    bird_path = os.path.join(source_dir, bird_id)
    if not os.path.isdir(bird_path):
        continue

    all_frames = [f for f in os.listdir(bird_path) if os.path.isfile(os.path.join(bird_path, f))]

    # Randomly selects 50 frames for testing (might be repetitive considering we are already separating full videos for testing)
    random.shuffle(all_frames)
    test_frames = all_frames[:50]
    remaining_frames = all_frames[50:]

    # Splits remaining frames: 70% train, 30% val
    train_frames, val_frames = train_test_split(remaining_frames, test_size=0.3, random_state=42)

    train_bird_dir = os.path.join(train_dir, bird_id)
    val_bird_dir   = os.path.join(val_dir, bird_id)
    test_bird_dir  = os.path.join(test_dir, bird_id)
    os.makedirs(train_bird_dir, exist_ok=True)
    os.makedirs(val_bird_dir, exist_ok=True)
    os.makedirs(test_bird_dir, exist_ok=True)

    # Copy and map training frames
    for frame in train_frames:
        src_frame_path = os.path.join(bird_path, frame)
        dst_frame_path = os.path.join(train_bird_dir, frame)
        shutil.copy2(src_frame_path, dst_frame_path)
        frame_dataset_mapping[(bird_id, frame)] = "Training"

    # Copy and map validation frames
    for frame in val_frames:
        src_frame_path = os.path.join(bird_path, frame)
        dst_frame_path = os.path.join(val_bird_dir, frame)
        shutil.copy2(src_frame_path, dst_frame_path)
        frame_dataset_mapping[(bird_id, frame)] = "Validation"

    # Copy and map testing frames
    for frame in test_frames:
        src_frame_path = os.path.join(bird_path, frame)
        dst_frame_path = os.path.join(test_bird_dir, frame)
        shutil.copy2(src_frame_path, dst_frame_path)
        frame_dataset_mapping[(bird_id, frame)] = "Testing"

    with open(log_file, "a") as f:
        f.write(f"{bird_id}\n")

    summary[bird_id] = {
        "total_frames": len(all_frames),
        "train_frames": len(train_frames),
        "val_frames": len(val_frames),
        "test_frames": len(test_frames),
    }
    print(f"Done processing bird '{bird_id}'")

print("\nSummary of processed bird folders:")
for bird_id, counts in summary.items():
    print(f"Bird ID: {bird_id}")
    print(f"  Total frames: {counts['total_frames']}")
    print(f"  Training frames: {counts['train_frames']}")
    print(f"  Validation frames: {counts['val_frames']}")
    print(f"  Testing frames: {counts['test_frames']}")
print("Done!")

# adds "Data set" column to the metadata file
metadata_df = pd.read_csv(metadata_file)

def lookup_dataset(row):
    # Use Bird ID and Frame Name to lookup dataset assignment
    key = (str(row["Bird ID"]).strip(), str(row["Frame Name"]).strip())
    return frame_dataset_mapping.get(key, "Unknown")  # Assign "Unknown" if not found

metadata_df["Data set"] = metadata_df.apply(lookup_dataset, axis=1)
metadata_df.to_csv(new_metadata_file, index=False)