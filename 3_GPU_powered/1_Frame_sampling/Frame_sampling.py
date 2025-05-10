# This script periodically extracts frames from videos, reserving a percentage
# of videos for testing (and skipping frame extraction from these videos).
# Extracted frames are saved along with metadata

import os
import random
import cv2
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = "/gpfs/data/fs72607/juarezs98/bbird_original_data/"
FRAME_OUTPUT_DIR = "/gpfs/data/fs72607/juarezs98/extracted_frames"
os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

METADATA_FILE = os.path.join(FRAME_OUTPUT_DIR, "metadata.csv")
TEST_METADATA_FILE = os.path.join(FRAME_OUTPUT_DIR, "test_videos.csv")
SAMPLING_INTERVAL = 240 # every 4 seconds (26 fps videos)
RANDOM_SEED = 42
TEST_RATIO = 0.1  # Ratio of test videos (10% of videos are reserved for testing)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if os.path.exists(METADATA_FILE): # loads metadata file or creates it
    metadata = pd.read_csv(METADATA_FILE)
else:
    metadata = pd.DataFrame(columns=["Bird ID", "Video Name", "Frame Name", "Timestamp (s)"])

if os.path.exists(TEST_METADATA_FILE):
    test_metadata = pd.read_csv(TEST_METADATA_FILE)
else:
    test_metadata = pd.DataFrame(columns=["Video Name"])

all_videos = [video.name for bird in Path(INPUT_DIR).iterdir() if bird.is_dir() for video in bird.glob("*.MP4")] 

if test_metadata.empty: # if test videos have not been selected, select them
    random.seed(RANDOM_SEED)
    test_metadata = pd.DataFrame(random.sample(all_videos, int(len(all_videos) * TEST_RATIO)), columns=["Video Name"])
    test_metadata.to_csv(TEST_METADATA_FILE, index=False)

test_videos = set(test_metadata["Video Name"])

for bird_folder in tqdm([b for b in Path(INPUT_DIR).iterdir() if b.is_dir()], desc="Processing videos"):
    bird_name = bird_folder.name
    for video_file in bird_folder.glob("*.MP4"):
        if video_file.name in test_videos:
            logging.info(f"Skipping test video: {video_file.name}")
            continue
        if video_file.name in metadata["Video Name"].unique():
            logging.info(f"Skipping already processed video: {video_file.name}")
            continue

        logging.info(f"Processing video: {video_file.name}")
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            logging.warning(f"Skipping corrupted video: {video_file.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps == 0:
            fps = 1  # Avoid division by zero

        frame_dir = os.path.join(FRAME_OUTPUT_DIR, bird_name)
        os.makedirs(frame_dir, exist_ok=True)
        frame_count = 0
        video_metadata = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % SAMPLING_INTERVAL == 0:
                frame_name = f"{video_file.stem}_frame{frame_count}.png"
                frame_path = os.path.join(frame_dir, frame_name)
                if cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                    video_metadata.append([bird_name, video_file.name, frame_name, frame_count / fps])
                    logging.info(f"Saved frame: {frame_path}")
            frame_count += 1
        cap.release()

        if len(video_metadata) > 0:
            new_metadata = pd.DataFrame(video_metadata, columns=metadata.columns)
            metadata = pd.concat([metadata, new_metadata], ignore_index=True)
            metadata.to_csv(METADATA_FILE, index=False)

logging.info("Done!")
