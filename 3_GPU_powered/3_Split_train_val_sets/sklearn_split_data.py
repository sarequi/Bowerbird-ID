import os
from sklearn.model_selection import train_test_split
import shutil

source_dir = "/gpfs/data/fs72607/juarezs98/masked_frames/"
train_dir = "/gpfs/data/fs72607/juarezs98/train_val_data/Training"
val_dir = "/gpfs/data/fs72607/juarezs98/train_val_data/Validation"
log_file = "processed_bird_ids.log"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Read already processed bird IDs from the log file
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        processed_bird_ids = set(line.strip() for line in f)
else:
    processed_bird_ids = set()

summary = {}

for item in os.listdir(source_dir):
    item_path = os.path.join(source_dir, item)
    
    # Check if the item is a directory - if so skip it!
    if not os.path.isdir(item_path):
        print(f"Skipping '{item}': Not a directory.")
        continue

    # Check if the directory has already been processed
    if item in processed_bird_ids:
        print(f"Skipping already processed bird '{item}'")
        continue

    # List all files in the bird's directory
    images = [img for img in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, img))]
    num_images = len(images)
    print(f"Found {num_images} images for bird '{item}'.")

    # Split images into training and validation
    train_imgs, val_imgs = train_test_split(images, test_size=0.3, random_state=42)
    print(f" - Training images: {len(train_imgs)}")
    print(f" - Validation images: {len(val_imgs)}")

    train_bird_dir = os.path.join(train_dir, item)
    val_bird_dir = os.path.join(val_dir, item)
    os.makedirs(train_bird_dir, exist_ok=True)
    os.makedirs(val_bird_dir, exist_ok=True)

    try:
        # Copy training images
        for img in train_imgs:
            source_img_path = os.path.join(item_path, img)
            dest_train_path = os.path.join(train_bird_dir, img)
            if not os.path.exists(dest_train_path):  # Skip if already copied
                shutil.copy2(source_img_path, dest_train_path)

        # Copy validation images
        for img in val_imgs:
            source_img_path = os.path.join(item_path, img)
            dest_val_path = os.path.join(val_bird_dir, img)
            if not os.path.exists(dest_val_path):  # Skip if already copied
                shutil.copy2(source_img_path, dest_val_path)

        # Log the processed bird ID
        with open(log_file, "a") as f:
            f.write(f"{item}\n")

        # Update summary
        summary[item] = {
            "training": len(train_imgs),
            "validation": len(val_imgs),
        }

        print(f"Done with bird '{item}'")

    except Exception as e:
        print(f"Error processing bird '{item}': {e}")

print("\nSummary of processed bird folders:")
for bird_id, counts in summary.items():
    print(f"Bird ID: {bird_id}")
    print(f"  Training images: {counts['training']}")
    print(f"  Validation images: {counts['validation']}")
print("Dataset successfully split into training and validation sets.")
