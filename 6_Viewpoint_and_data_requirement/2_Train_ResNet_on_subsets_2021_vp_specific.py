'''
Trains multiple ResNet50 models, for each viewpoint (front, back, left_side, right_side, side_view) and for 
a range of subset sizes (50-1000). For every (viewpoint, subset_size) combination, the script:

1. trains a classifier,
2. Records the best validation accuracy,
3. Saves the best performing model, and 
4. Creates CSV + plots of accuracy vs subset size, and an overall summary plot across viewpoints
'''

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

DATA_ROOT   = Path("4.4_Viewpoint_subsets/Data")
OUTPUT_ROOT = Path("4.4_Viewpoint_subsets/training_outputs")
VIEWPOINTS  = ["front", "back", "left_side", "right_side", "side_view"]
SUBSET_SIZES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# training params
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
STEP_SIZE = 7
GAMMA = 0.1
NUM_WORKERS = 8

print("Using device:", DEVICE)

# transforms (fix: use weights.transforms() pipeline directly)
weights = ResNet50_Weights.IMAGENET1K_V2
base_tf = weights.transforms()  # includes ToTensor + Normalize
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    base_tf,
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    base_tf,
])

def has_enough_classes(dir_path: Path) -> bool:
    if not dir_path.exists():
        return False
    class_dirs = [p for p in dir_path.iterdir() if p.is_dir()]
    if len(class_dirs) < 2:
        return False
    for cd in class_dirs:
        any_img = any(f.suffix.lower() in {".jpg", ".jpeg", ".png"} for f in cd.iterdir() if f.is_file())
        if not any_img:
            return False
    return True

def train_one_viewpoint(subset_size: int, vp: str) -> Tuple[bool, float]:
    subset_dir = DATA_ROOT / str(subset_size) / vp
    train_dir = subset_dir / "train"
    val_dir   = subset_dir / "val"

    if not (has_enough_classes(train_dir) and has_enough_classes(val_dir)):
        return False, 0.0

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_ds), "val": len(val_ds)}
    class_names = train_ds.classes

    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_acc = 0.0
    best_state = None

    for _ in range(NUM_EPOCHS):
        for phase in ["train", "val"]:
            model.train(mode=(phase == "train"))
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

            if phase == "train":
                scheduler.step()

            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_state = model.state_dict()

    out_dir = OUTPUT_ROOT / str(subset_size) / vp
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, out_dir / "best_model.pth")
    (out_dir / "classes.txt").write_text("\n".join(class_names))

    return True, best_acc

# run training
results: Dict[str, List[Tuple[int, float]]] = {vp: [] for vp in VIEWPOINTS}

for vp in VIEWPOINTS:
    print(f"\n=== Viewpoint: {vp} ===")
    for size in SUBSET_SIZES:
        trained, best_acc = train_one_viewpoint(size, vp)
        if trained:
            results[vp].append((size, best_acc))
            print(f"  size {size:>4}: {best_acc:.4f}")
        else:
            print(f"  size {size:>4}: skipped")

# per-viewpoint CSV + plot
summ_dir = OUTPUT_ROOT / "summaries"
summ_dir.mkdir(parents=True, exist_ok=True)

for vp, entries in results.items():
    if not entries:
        continue
    df = pd.DataFrame(entries, columns=["subset_size", "best_val_accuracy"])
    df.to_csv(summ_dir / f"{vp}_acc_by_subset.csv", index=False)

    plt.figure()
    plt.plot(df["subset_size"], df["best_val_accuracy"], marker="o")
    plt.title(f"{vp}")
    plt.xlabel("subset size")
    plt.ylabel("val acc")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(summ_dir / f"{vp}_acc_by_subset.png", dpi=150, bbox_inches="tight")
    plt.close()

# combined table and plot
from collections import defaultdict

acc_map = defaultdict(dict)  # {size: {vp: acc}}
for vp, entries in results.items():
    for size, acc in entries:
        acc_map[size][vp] = acc

all_sizes = sorted(acc_map.keys())
wide = {"subset_size": all_sizes}
for vp in VIEWPOINTS:
    wide[vp] = [acc_map[s].get(vp, np.nan) for s in all_sizes]

df_all = pd.DataFrame(wide)
df_all.to_csv(summ_dir / "overall_accuracy_by_subset.csv", index=False)

plt.figure(figsize=(8, 5))
for vp in VIEWPOINTS:
    xs = [s for s in all_sizes if not np.isnan(acc_map[s].get(vp, np.nan))]
    ys = [acc_map[s][vp] for s in xs]
    if xs:
        plt.plot(xs, ys, marker="o", label=vp)

plt.title("Accuracy across subset sizes (by viewpoint)")
plt.xlabel("subset size")
plt.ylabel("val acc")
plt.ylim(0, 1.0)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="viewpoint")
plt.savefig(summ_dir / "overall_accuracy_by_subset.png", dpi=150, bbox_inches="tight")
plt.close()

print("Done")
