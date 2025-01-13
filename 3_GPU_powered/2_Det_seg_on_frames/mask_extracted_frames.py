import os
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import label

input_dir = "/gpfs/data/fs72607/juarezs98/extracted_frames"
input_metadata_csv = "/gpfs/data/fs72607/juarezs98/extracted_frames/metadata.csv"
output_dir = "/gpfs/data/fs72607/juarezs98/masked_frames"
new_metadata_csv = "/gpfs/data/fs72607/juarezs98/masked_frames/masked_frames_metadata.csv"

yolo_version = 'x'
yolo_model = YOLO(f'yolo11{yolo_version}-seg.pt')

MIN_BLOB_PIXELS = 5000
BOTTOM_FRACTION_ROW = 1 / 5  # Fraction of the row to filter during for horizontal row filtering
BOTTOM_FRACTION_NARROW = 1 / 3  # Bottom of the image to filter during narrow structure filtering
HORIZONTAL_THRESHOLD = 0.8  # Removes rows with >=80% black pixels 
NARROW_SEGMENT_THRESHOLD = 100  # Maximum width of narrow structures to remove

os.makedirs(output_dir, exist_ok=True)

metadata = pd.read_csv(input_metadata_csv)

if os.path.exists(new_metadata_csv):
    processed_frames = pd.read_csv(new_metadata_csv)["Frame Name"].tolist()  # Check existing frames
else:
    processed_frames = []

if not os.path.exists(new_metadata_csv): # Initialises a new metadata file IF there is none
    new_metadata_df = pd.DataFrame(columns=["Bird ID", "Video Name", "Frame Name", "Timestamp (s)", "Masked Image"])
    new_metadata_df.to_csv(new_metadata_csv, index=False)

print("Processing metadata entries")
for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing metadata"):
    row = row.to_dict() # Converts row to a dictionary to avoid NumPy indexing issues (ERROR)

    bird_id = row["Bird ID"]
    frame_name = row["Frame Name"]

    if frame_name in processed_frames: # Skips frame if it was already processed AND exists in the metadata
        print(f"Skipping frame {frame_name} as it is already in the metadata.")
        continue

    frame_path = os.path.join(input_dir, bird_id, frame_name)

    if not os.path.exists(frame_path):  # Also skips frames that dont exist
        continue

    bird_mask_dir = os.path.join(output_dir, bird_id)
    os.makedirs(bird_mask_dir, exist_ok=True)

    # YOLO prediction
    results = yolo_model.predict(frame_path, conf=0.6, verbose=False)
    if len(results[0].boxes) == 0:  # Skip frame if there are no detections
        continue

    img = results[0].orig_img
    x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])  # Get highest-confidence box
    cropped = img[y1:y2, x1:x2]

    # Mask prediction
    results = yolo_model.predict(cropped, conf=0.8, verbose=False)
    if results[0].masks is None:  # Skip frame if there are no masks
        continue

    mask = results[0].masks.data[0].cpu().numpy().astype(bool)

    # Ensures the mask dimensions match the cropped image dimensions
    if mask.shape[:2] != cropped.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (cropped.shape[1], cropped.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    # Horizontal row filtering
    bottom_start_row = int(mask.shape[0] * (1 - BOTTOM_FRACTION_ROW))
    bottom_mask_row = mask[bottom_start_row:, :] 

    for row_idx in range(bottom_mask_row.shape[0]):
        row_pixels = bottom_mask_row[row_idx, :]
        black_percentage = 1 - (np.sum(row_pixels) / row_pixels.shape[0])  # Percentage of black pixels

        if black_percentage >= HORIZONTAL_THRESHOLD:  # >= 80% black pixels
            bottom_mask_row[row_idx, :] = 0  # Black out the row

    mask[bottom_start_row:, :] = bottom_mask_row # Replace the modified bottom mask back into the original 

    # Narrow structure filtering 
    bottom_start_narrow = int(mask.shape[0] * (1 - BOTTOM_FRACTION_NARROW))
    bottom_mask_narrow = mask[bottom_start_narrow:, :]  # Filter narrow structures in the bottom fraction of the frame

    for row_idx in range(bottom_mask_narrow.shape[0]):
        row_pixels = bottom_mask_narrow[row_idx, :]
        non_black_segments = np.split(np.where(row_pixels)[0], np.where(np.diff(np.where(row_pixels)[0]) > 1)[0] + 1)

        for segment in non_black_segments:
            if len(segment) <= NARROW_SEGMENT_THRESHOLD:  # Narrow segment
                row_pixels[segment] = 0  # Turn the narrow segment black (we assume narrow segments are legs)

        bottom_mask_narrow[row_idx, :] = row_pixels  # Update the row after removing narrow segments

    mask[bottom_start_narrow:, :] = bottom_mask_narrow # Replace the modified narrow mask back into the original 

    # Remove small blobs
    labeled_mask, num_features = label(mask)
    filtered_mask = np.zeros_like(mask, dtype=bool)
    for region_label in range(1, num_features + 1):
        if np.sum(labeled_mask == region_label) >= MIN_BLOB_PIXELS:
            filtered_mask[labeled_mask == region_label] = True

    # Skip this frame if no blob is larger than the minimum blob size
    if not np.any(filtered_mask):
        continue

    # Ensure filtered_mask dimensions match cropped dimensions
    if filtered_mask.shape[:2] != cropped.shape[:2]:
        filtered_mask = cv2.resize(filtered_mask.astype(np.uint8), (cropped.shape[1], cropped.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

    # Save masked image
    mask_rgb = np.zeros_like(cropped)
    mask_rgb[filtered_mask] = cropped[filtered_mask]
    masked_frame_name = f"{os.path.splitext(frame_name)[0]}_mask.png"
    mask_path = os.path.join(bird_mask_dir, masked_frame_name)

    if not os.path.exists(mask_path): # Skip writing the masked image if it already exists (ERROR)
        cv2.imwrite(mask_path, mask_rgb)

    # Create new entry for the metadata file
    new_entry = {
        "Bird ID": bird_id,
        "Video Name": row['Video Name'],
        "Frame Name": frame_name,
        "Timestamp (s)": row['Timestamp (s)'],
        "Masked Image": masked_frame_name
    }

    # Append the new entry to the file
    pd.DataFrame([new_entry]).to_csv(new_metadata_csv, mode='a', header=False, index=False)

    # Add the frame to the processed frames list (to avoid reprocessing)
    processed_frames.append(frame_name)

print(f"Metadata updated at: {new_metadata_csv}")




