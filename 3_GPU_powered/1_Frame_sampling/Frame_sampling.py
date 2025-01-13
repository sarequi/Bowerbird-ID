import os
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd
import logging

INPUT_DIR = "/gpfs/data/fs72607/juarezs98/bbird_original_data/"
FRAME_OUTPUT_DIR = "/gpfs/data/fs72607/juarezs98/extracted_frames"
METADATA_FILE = os.path.join(FRAME_OUTPUT_DIR, "metadata.csv")
YOLO_MODEL_PATH = "yolo11m-seg.pt"
SAMPLING_INTERVAL = 240
MAX_FRAMES_PER_VIDEO = 10
SIMILARITY_THRESHOLD = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def ensure_dir_exists(directory):
    os.makedirs(directory, exist_ok=True)

def load_model(model_path):
    """Load YOLO model with GPU support"""
    model = YOLO(model_path)
    return model.to("cuda").eval()

def load_or_create_metadata(metadata_file):
    """Load existing metadata or create an empty one"""
    if os.path.exists(metadata_file):
        metadata = pd.read_csv(metadata_file)
        if metadata.empty or list(metadata.columns) != ["Bird ID", "Video Name", "Frame Name", "Timestamp (s)"]:
            logging.warning(f"Metadata file is empty or has incorrect columns. Recreating it: {metadata_file}")
            metadata = pd.DataFrame(columns=["Bird ID", "Video Name", "Frame Name", "Timestamp (s)"])
            metadata.to_csv(metadata_file, index=False)
    else:
        logging.info(f"Metadata file not found. Creating new metadata file: {metadata_file}")
        metadata = pd.DataFrame(columns=["Bird ID", "Video Name", "Frame Name", "Timestamp (s)"])
        metadata.to_csv(metadata_file, index=False)
    return metadata

def delete_frames_except(metadata, bird_name, video_name):
    """Delete all frames not listed in the metadata for the current video"""
    video_frames = metadata[(metadata["Bird ID"] == bird_name) & (metadata["Video Name"] == video_name)]["Frame Name"]
    frame_dir = os.path.join(FRAME_OUTPUT_DIR, bird_name)
    for frame_file in Path(frame_dir).glob(f"{Path(video_name).stem}_frame*.png"):
        if frame_file.name not in video_frames.values:
            os.remove(frame_file)
            logging.info(f"Deleted intermediate frame: {frame_file}")

# Step 1: Process a single video
def process_video(video_path, bird_name, metadata_file, model):
    """Process a single video and keep up to 10 unique frames"""
    logging.info(f"Processing video: {video_path}")
    
    # Step 1.1: Extract frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Skipping corrupted or unreadable video: {video_path}")
        return  # Skip corrupted video

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        logging.warning(f"Invalid FPS for video: {video_path}")
        return  # Skip if FPS is invalid

    metadata = load_or_create_metadata(metadata_file)
    video_metadata = []

    frame_dir = os.path.join(FRAME_OUTPUT_DIR, bird_name)
    ensure_dir_exists(frame_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % SAMPLING_INTERVAL == 0:
            frame_name = f"{Path(video_path).stem}_frame{frame_count}.png"
            frame_path = os.path.join(frame_dir, frame_name)

            success = cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if success:
                timestamp = frame_count / fps
                video_metadata.append([bird_name, video_path.name, frame_name, timestamp])
                logging.info(f"Saved frame: {frame_path}")
        frame_count += 1

    cap.release()

    # Append extracted frames to metadata
    if video_metadata:
        new_metadata = pd.DataFrame(video_metadata, columns=metadata.columns)
        metadata = pd.concat([metadata, new_metadata], ignore_index=True)
        metadata.to_csv(metadata_file, index=False)
        logging.info(f"Appended {len(new_metadata)} new frames to metadata.")
    else:
        logging.warning(f"No frames were extracted for video: {video_path}")
        return

# Step 2: Process all videos
def process_all_videos(input_dir, metadata_file, model):
    """Process all videos sequentially"""
    metadata = load_or_create_metadata(metadata_file)
    bird_folders = [f for f in Path(input_dir).iterdir() if f.is_dir()]

    for bird_folder in tqdm(bird_folders, desc="Processing videos"):
        bird_name = bird_folder.name
        for video_file in bird_folder.glob("*.MP4"):
            if video_file.name in metadata["Video Name"].unique():
                logging.info(f"Skipping already processed video: {video_file.name}")
                continue

            logging.info(f"Processing video: {video_file.name}")
            process_video(video_file, bird_name, metadata_file, model)

def main():
    ensure_dir_exists(FRAME_OUTPUT_DIR)

    logging.info("Starting video processing")
    try:
        model = load_model(YOLO_MODEL_PATH)
        process_all_videos(INPUT_DIR, METADATA_FILE, model)
        logging.info("Done!")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
