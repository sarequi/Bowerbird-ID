import os
import shutil
import pandas as pd
import re
from tqdm import tqdm

metadata_file = "/gpfs/data/fs72607/juarezs98/extracted_frames/test_videos.csv"
source_directory = "/gpfs/data/fs72607/juarezs98/bbird_original_data/"
destination_directory = "/gpfs/data/fs72607/juarezs98/extracted_frames/Extracted_test_videos"

os.makedirs(destination_directory, exist_ok=True)
metadata = pd.read_csv(metadata_file)

# Iterate over the metadata file
for video_name in tqdm(metadata["Video Name"]):
    match = re.match(r"(B\d+)_", video_name)  # Extract Bird ID from video name
    if match:
        bird_id = match.group(1) 
        bird_folder = os.path.join(source_directory, bird_id)  # Construct folder path
        
        if os.path.exists(bird_folder) and os.path.isdir(bird_folder):  # Check if folder exists
            for file in os.listdir(bird_folder):  # Iterate through videos in the folder
                if file == video_name:  # Check for a matching video
                    source_path = os.path.join(bird_folder, file)
                    destination_path = os.path.join(destination_directory, file)
                    shutil.copy2(source_path, destination_path)  # Copy the file
        else:
            print(f"Folder not found: {bird_folder}")
    else:
        print(f"No Bird ID found in filename {video_name}") 