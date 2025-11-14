#!/usr/bin/env python3

# Runs the viewpoint classifier on all non-test frames and creates a CSV of viewpoint predictions

import os
import csv
import sys
from pathlib import Path
from PIL import Image, ImageFile

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18
from torchvision.datasets.folder import is_image_file

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

ROOT_DIR = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/non_test_frames")
WEIGHTS  = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/trained_vp_clssfr/resnet18_viewpoint_best.pth")
OUT_CSV  = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/vp_predictions_full_dataset/non_test_predicitons.csv")
DEFAULT_LABELS = ["back", "front", "left_side", "right_side"]
BATCH_SIZE = 256

def log(msg: str):
    print(msg, flush=True)

def load_labels(weights_path: Path):
    j = weights_path.parent / "class_to_idx.json"
    if j.exists():
        import json
        mapping = json.loads(j.read_text())
        inv = {v: k for k, v in mapping.items()}
        return [inv[i] for i in range(len(inv))]
    return DEFAULT_LABELS

class ImagePaths(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted([p for p in root.rglob("*") if p.is_file() and is_image_file(p.name)])
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.tf = T.Compose([
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        self.bad = 0

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            return self.tf(img), str(p)
        except Exception:
            # Mark as bad; return a tiny dummy to keep the batch shape
            self.bad += 1
            # caller will drop this row by writing empty path
            dummy = torch.zeros(3, 224, 224)
            return dummy, ""

def build_model(num_classes: int, device: torch.device):
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.to(device)
    m.eval()
    return m

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels(WEIGHTS)
    num_classes = len(labels)

    log(f"Scanning {ROOT_DIR} ...")
    ds = ImagePaths(ROOT_DIR)
    total = len(ds)
    log(f"Found {total} images.")
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "8"))
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type=="cuda"),
        persistent_workers=(num_workers > 0), prefetch_factor=2
    )

    model = build_model(num_classes, device)
    log(f"Loading weights from {WEIGHTS}")
    ckpt = torch.load(WEIGHTS, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)

    softmax = nn.Softmax(dim=1)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_rows = 0
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "pred_label", "prob"])
        log(f"Writing to {OUT_CSV}")

        for i, (x, paths) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x)
            probs = softmax(logits)
            conf, pred_idx = probs.max(dim=1)

            # write only valid rows (path != "")
            batch_written = 0
            for pth, idx, pr in zip(paths, pred_idx.tolist(), conf.tolist()):
                if not pth:
                    skipped_rows += 1
                    continue
                writer.writerow([pth, labels[idx], f"{pr:.6f}"])
                batch_written += 1
            written += batch_written
            f.flush()  # make progress visible on disk

            if i % 5 == 0 or i == 1:
                log(f"Processed {min(i*BATCH_SIZE, total)} / {total} "
                    f"(written {written}, skipped {skipped_rows}, bad imgs seen {ds.bad})")

    log(f"Done. Total written: {written}, skipped rows: {skipped_rows}, bad images: {ds.bad}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        sys.exit(1)
