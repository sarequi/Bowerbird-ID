import os
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
from PIL import Image
import imagehash

# Frame Extraction

INPUT_DIR = "home/fs72607/juarezs98/Bowerbird-ID/7_Classify_bowerdbird_ID/Videos_to_classify"
OUTPUT_DIR = "home/fs72607/juarezs98/Bowerbird-ID/7_Classify_bowerdbird_ID/Extracted_frames"
YOLO_MODEL_PATH = "yolo11x-seg.pt"
SAMPLING_INTERVAL = 60  # Extract a frame every X frames
IOU_THRESHOLD = 0.5     # threshold for filtering overlapping detections
SIMILARITY_THRESHOLD = 5  # pHash similarity threshold

yolo_model = YOLO(YOLO_MODEL_PATH)  # Load YOLO model


def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2  

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    unique_hashes = []
    multi_bird_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract every SAMPLING_INTERVAL-th frame
        if frame_count % SAMPLING_INTERVAL == 0:
            results = yolo_model.predict(frame, conf=0.6, verbose=False)
            detections = results[0].boxes

            if len(detections) > 0:
                # Filter detections based on IoU and pick the highest-confidence bbox
                filtered_detections = []
                for i, box in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    score = detections.conf[i]
                    # Only add if IoU with existing detections is below threshold
                    if all(calculate_iou((x1, y1, x2, y2), det[:4]) <= IOU_THRESHOLD for det in filtered_detections):
                        filtered_detections.append((x1, y1, x2, y2, score))

                # If more than one detection remains, we suspect multiple birds
                if len(filtered_detections) > 1:
                    multi_bird_detected = True
                    break

                # Grab the detection with the highest confidence
                x1, y1, x2, y2, _ = max(filtered_detections, key=lambda d: d[-1])
                cropped = frame[y1:y2, x1:x2]

                # Compute a perceptual hash of the cropped image
                pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                frame_hash = imagehash.phash(pil_image)

                # Save frame only if it's not too similar to previous frames
                if all(abs(frame_hash - h) > SIMILARITY_THRESHOLD for h in unique_hashes):
                    frame_name = f"{video_path.stem}_frame{frame_count}.png"
                    frame_path = os.path.join(OUTPUT_DIR, frame_name)
                    cv2.imwrite(frame_path, frame)
                    unique_hashes.append(frame_hash)

        frame_count += 1

    cap.release()

    if multi_bird_detected:
        print(f"There is more than one bird in the video {video_path.name}, better choose another one!")
    else:
        print(f"Processed video {video_path.name}")


video_files = list(Path(INPUT_DIR).glob("*.MP4"))
if not video_files:
    print("No videos found in the input directory")
else:
    for video in tqdm(video_files, desc="Processing videos"):
        process_video(video)

print(f"Extracted frames saved in: {OUTPUT_DIR}")


# Mask Processing

from scipy.ndimage import label

input_dir = "extracted_frames"   # Directory holding extracted frames
yolo_model = YOLO("yolo11x-seg.pt")  # Load the YOLO segmentation model again

MIN_BLOB_PIXELS = 5000
BOTTOM_FRACTION_ROW = 1 / 4
BOTTOM_FRACTION_NARROW = 1 / 2
HORIZONTAL_THRESHOLD = 0.8
NARROW_SEGMENT_THRESHOLD = 100


def filter_horizontal_rows(mask, threshold, bottom_fraction):
    """
    Filters out horizontal rows near the bottom of the image if
    the fraction of black pixels is above a certain threshold.
    """
    start_row = int(mask.shape[0] * (1 - bottom_fraction))
    for row_idx in range(start_row, mask.shape[0]):
        row = mask[row_idx, :]
        # If the row is mostly black (above threshold), clear it
        if 1 - (np.sum(row) / row.shape[0]) >= threshold:
            mask[row_idx, :] = 0
    return mask


def filter_narrow_segments(mask, max_width, bottom_fraction):
    """
    Filters out narrow segments of white pixels in each row
    near the bottom of the image if their width is under a threshold.
    """
    start_row = int(mask.shape[0] * (1 - bottom_fraction))
    for row_idx in range(start_row, mask.shape[0]):
        indices = np.where(mask[row_idx])[0]
        segments = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        for segment in segments:
            if len(segment) <= max_width:
                mask[row_idx, segment] = 0
    return mask


def remove_small_blobs(mask, min_pixels):
    """
    Removes connected blobs smaller than a certain pixel count.
    """
    labeled_mask, num_features = label(mask)
    # Keep only blobs that have at least 'min_pixels' pixels
    valid_labels = [
        i for i in range(1, num_features + 1)
        if np.sum(labeled_mask == i) >= min_pixels
    ]
    return np.isin(labeled_mask, valid_labels)


for frame_name in tqdm(os.listdir(input_dir), desc="Frames"):
    frame_path = os.path.join(input_dir, frame_name)
    if not frame_name.lower().endswith('.png'):
        continue

    # Run YOLO detection on the full frame
    results = yolo_model.predict(frame_path, conf=0.4, verbose=False)
    if not results[0].boxes:
        os.remove(frame_path)
        continue

    # Crop region of interest using the highest-confidence box
    img = cv2.imread(frame_path)
    x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
    cropped = img[y1:y2, x1:x2]

    # Predict segmentation mask on cropped region
    mask_results = yolo_model.predict(cropped, conf=0.6, verbose=False)
    if not mask_results[0].masks:
        os.remove(frame_path)
        continue

    mask = mask_results[0].masks.data[0].cpu().numpy().astype(bool)

    # Apply mask filters
    mask = filter_horizontal_rows(mask, HORIZONTAL_THRESHOLD, BOTTOM_FRACTION_ROW)
    mask = filter_narrow_segments(mask, NARROW_SEGMENT_THRESHOLD, BOTTOM_FRACTION_NARROW)
    mask = remove_small_blobs(mask, MIN_BLOB_PIXELS)

    # If no part of the mask remains, discard the frame
    if not np.any(mask):
        os.remove(frame_path)
        continue

    # Resize mask if needed
    if mask.shape[:2] != cropped.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (cropped.shape[1], cropped.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    # Apply mask to the cropped image
    mask_rgb = np.zeros_like(cropped)
    mask_rgb[mask] = cropped[mask]
    cv2.imwrite(frame_path, mask_rgb)


# Classification

import torch
from torchvision import models, transforms
from collections import Counter

MODEL_PATH = "home/fs72607/juarezs98/Bowerbird-ID/6_Train_ResNet50/best_model.pth"
FRAME_DIR = "extracted_frames"

CLASS_NAMES = [
    'B02', 'B03', 'B04', 'B05', 'B07', 'B11', 'B18', 'B23',
    'B26', 'B29', 'B30', 'B31', 'B47', 'B49', 'B50', 'B52'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with 16 output classes
model = models.resnet50(pretrained=False, num_classes=16)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

predictions = []
for frame_name in tqdm(os.listdir(FRAME_DIR), desc="Processing frames"):
    frame_path = os.path.join(FRAME_DIR, frame_name)
    if not frame_name.lower().endswith('.png'):
        continue

    image = Image.open(frame_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = probabilities.argmax().item()

        if predicted_class >= len(CLASS_NAMES):
            print(f"Warning: Predicted class index {predicted_class} is out of bounds for CLASS_NAMES.")
            continue

        predictions.append(predicted_class)

# Count how many times each class was predicted
prediction_counts = Counter(predictions)
total_predictions = sum(prediction_counts.values())

if total_predictions == 0:
    print("No valid predictions were made.")
    exit()

# Calculate percentage for each class
percentages = {
    CLASS_NAMES[k]: (v / total_predictions) * 100
    for k, v in prediction_counts.items()
}
sorted_percentages = dict(
    sorted(percentages.items(), key=lambda item: item[1], reverse=True)
)

# Determine the most common class
most_common_class = max(prediction_counts, key=prediction_counts.get)
most_common_percentage = sorted_percentages[CLASS_NAMES[most_common_class]]

# Print results
print("\nPrediction Results:")
for bird_id, percentage in sorted_percentages.items():
    print(f"{bird_id}: {percentage:.2f}%")

print(f"\nThe bird is most likely {CLASS_NAMES[most_common_class]}")
