import os
from sklearn.model_selection import train_test_split
import shutil

source_dir = "/gpfs/data/fs72607/juarezs98/masked_frames/"
train_dir = "/gpfs/data/fs72607/juarezs98/train_val_data/Training"
val_dir = "/gpfs/data/fs72607/juarezs98/train_val_data/Validation"
log_file = "processed_bird_ids.log"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Read already processed bird IDs from log (useful if there is a need to re run the script)
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        processed_bird_ids = set(line.strip() for line in f)
else:
    processed_bird_ids = set()

summary = {}

# Iterate through each bird ID folder
for bird_id in os.listdir(source_dir):
    if bird_id in processed_bird_ids:
        print(f"Skipping already processed bird '{bird_id}'")
        continue

    bird_path = os.path.join(source_dir, bird_id)
    if os.path.isdir(bird_path):
        images = os.listdir(bird_path)

        if not images:  # Print a warning if the folder is empty
            print(f"No images found for bird '{bird_id}' in {bird_path}")
            continue

        # Split images into training and validation sets
        train_imgs, val_imgs = train_test_split(images, test_size=0.3, random_state=42)

        train_bird_dir = os.path.join(train_dir, bird_id)
        val_bird_dir = os.path.join(val_dir, bird_id)
        os.makedirs(train_bird_dir, exist_ok=True)
        os.makedirs(val_bird_dir, exist_ok=True)

        try:
            # Copy training images
            for img in train_imgs:
                source_img_path = os.path.join(bird_path, img)
                dest_train_path = os.path.join(train_bird_dir, img)
                if not os.path.exists(dest_train_path):  # Skip if already copied
                    shutil.copy2(source_img_path, dest_train_path)

            # Copy validation images
            for img in val_imgs:
                source_img_path = os.path.join(bird_path, img)
                dest_val_path = os.path.join(val_bird_dir, img)
                if not os.path.exists(dest_val_path):  # Skip if already copied
                    shutil.copy2(source_img_path, dest_val_path)

            with open(log_file, "a") as f:
                f.write(f"{bird_id}\n")

            # summary
            summary[bird_id] = {
                "training": len(train_imgs),
                "validation": len(val_imgs),
            }

            print(f"Done processing bird '{bird_id}'")

        except Exception as e:
            print(f"Error processing bird '{bird_id}': {e}")

print("\nSummary of processed bird folders:")
for bird_id, counts in summary.items():
    print(f"Bird ID: {bird_id}")
    print(f"  Training images: {counts['training']}")
    print(f"  Validation images: {counts['validation']}")
print("Done")
