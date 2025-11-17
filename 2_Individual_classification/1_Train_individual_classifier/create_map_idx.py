# make_class_map.py
import json, os
from pathlib import Path
from torchvision import datasets

TRAIN_DIR = "train"
MODEL_DIR = "training_outputs"

ds = datasets.ImageFolder(TRAIN_DIR)
mapping = ds.class_to_idx                 
classes = ds.classes                      

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
with open(os.path.join(MODEL_DIR, "class_to_idx.json"), "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)
with open(os.path.join(MODEL_DIR, "classes.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(classes))

print(f"Saved: {MODEL_DIR}/class_to_idx.json and classes.txt")