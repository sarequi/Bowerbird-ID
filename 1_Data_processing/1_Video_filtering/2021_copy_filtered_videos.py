from pathlib import Path
import pandas as pd
import shutil
from collections import defaultdict
from tqdm.auto import tqdm

raw_videos_root = Path("raw_videos_2021/") 
filtered_scoring_csv    = Path("2021_filtered_video_categorisation.csv")
sorted_videos_root   = Path("1_filtered_vids_2021_by_bird/")

# build Video_ID - bird map
df = pd.read_csv(filtered_scoring_csv, encoding="utf-16")
id_to_bird = {
    Path(video_id).stem: bird
    for video_id, bird in zip(df["Video_ID"].astype(str), df["Bird"].astype(str))
}

# create dir for each bird
sorted_videos_root.mkdir(parents=True, exist_ok=True)
for bird in set(id_to_bird.values()):
    (sorted_videos_root / bird).mkdir(parents=True, exist_ok=True)

# walk raw_videos_root and copy videos in the map
files_copied = 0
bird_counts = defaultdict(int)
seen_stems = set() # prevents duplicates

total_files = sum(1 for _ in raw_videos_root.rglob("*") if _.is_file())

for path in tqdm(raw_videos_root.rglob("*"), total=total_files):
    if not path.is_file():
        continue
    stem = path.stem
    bird = id_to_bird.get(stem)
    if bird is None or stem in seen_stems:
        continue
    dest_path = sorted_videos_root / bird / path.name
    if dest_path.exists(): # skip if already copied
        seen_stems.add(stem)
        continue
    try:
        shutil.copy2(path, dest_path)
        files_copied += 1
        bird_counts[bird] += 1
        seen_stems.add(stem)
    except Exception as e:
        tqdm.write(f"Skipping {path}: {e}")

print(f"Total videos copied: {files_copied}")
for bird, count in sorted(bird_counts.items()):
    print(f"Bird {bird}: {count} video(s)")