#!/usr/bin/env python3
import os, json, csv, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet50
from sklearn.metrics import classification_report as sk_classification_report

from tqdm.auto import tqdm

MODEL_DIR = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/2_individual_classifier/trained_individual_clssfr/all_viewpoints/training_outputs")
MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASS_MAP  = MODEL_DIR / "class_to_idx.json"
TEST_DIR   = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/test_frames")
OUT_DIR    = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/2_individual_classifier/test_on_test_set")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE  = 512
BATCH_SIZE  = 64
NUM_WORKERS = 8
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS    = {".png"}  # frames are png; add more if needed: {".png", ".jpg", ".jpeg", ".webp", ...}

# tqdm-friendly logger
def log(m: str): 
    tqdm.write(f"[eval] {m}")

# Transformation (no augments)
tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ImageFolder that ignores hidden/empty dirs & only valid image files
class CleanImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, ignore_hidden=True):
        self.ignore_hidden = ignore_hidden
        super().__init__(root, transform=transform, is_valid_file=self._is_valid)

    def find_classes(self, directory: str):
        classes = []
        for entry in os.scandir(directory):
            if not entry.is_dir():
                continue
            name = entry.name
            if self.ignore_hidden and name.startswith("."):
                continue
            # keep only folders that actually contain valid images
            has_img = False
            for _, _, files in os.walk(entry.path):
                if any(Path(f).suffix.lower() in IMG_EXTS for f in files):
                    has_img = True
                    break
            if has_img:
                classes.append(name)
        classes.sort()
        if not classes:
            raise FileNotFoundError("No class folders with images found.")
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def _is_valid(path: str) -> bool:
        return Path(path).suffix.lower() in IMG_EXTS

def rebuild_model(num_outputs:int)->nn.Module:
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_outputs)
    return m.to(DEVICE)

def main():
    t0 = time.time()

    # Load training mapping 
    if not CLASS_MAP.is_file():
        raise FileNotFoundError(f"Missing mapping: {CLASS_MAP}")
    with CLASS_MAP.open("r", encoding="utf-8") as f:
        train_map = {str(k): int(v) for k, v in json.load(f).items()}
    idx_to_name_train = {v: k for k, v in train_map.items()}
    num_train_classes = len(train_map)
    train_names_ordered = [idx_to_name_train[i] for i in range(num_train_classes)]
    log(f"TRAIN mapping: {num_train_classes} classes.")
    log(f"TRAIN order: {', '.join(train_names_ordered)}")

    # Load model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    model = rebuild_model(num_train_classes)
    state = torch.load(MODEL_PATH, map_location="cpu")
    state_to_load = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_to_load, strict=True)
    model.eval()

    # Build test dataset loader
    if not TEST_DIR.is_dir():
        raise NotADirectoryError(f"Test dir not found: {TEST_DIR}")
    test_ds = CleanImageFolder(str(TEST_DIR), transform=tf)
    test_classes = test_ds.classes
    log(f"TEST classes ({len(test_classes)}): {', '.join(test_classes)}")

    # Evaluate only overlapping classes
    subset_names = [c for c in test_classes if c in train_map]
    missing = sorted(set(test_classes) - set(subset_names))
    if missing:
        log(f"Skipping (not in training): {missing}")
    if not subset_names:
        raise RuntimeError("No overlap between TEST and TRAIN classes.")

    # Columns (logit indices) to keep from the model outputs
    subset_cols = torch.tensor([train_map[c] for c in subset_names], dtype=torch.long, device=DEVICE)

    # Map test label indices -> subset indices [0..K-1]
    test_idx_to_subset = {i: subset_names.index(name) if name in subset_names else None
                          for i, name in enumerate(test_classes)}

    loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )

    all_true, all_pred = [], []
    skipped_nonoverlap = 0

    with torch.inference_mode():
        pbar = tqdm(total=len(test_ds), unit="img", desc="Evaluating", dynamic_ncols=True)
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            logits = model(x)                                 # [B, num_train_classes]
            logits_sub = logits.index_select(1, subset_cols)  # [B, |subset|]
            pred_sub = torch.argmax(logits_sub, dim=1).cpu().tolist()

            # remap y (test indices) to subset indices; skip non-overlap
            for yy, pp in zip(y.tolist(), pred_sub):
                mapped = test_idx_to_subset.get(yy, None)
                if mapped is None:
                    skipped_nonoverlap += 1
                    continue
                all_true.append(mapped)
                all_pred.append(pp)

            pbar.update(x.size(0))
        pbar.close()

    # Metrics
    target_names = subset_names
    report = sk_classification_report(
        all_true, all_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
        digits=4
    )
    print("\nClassification report (TEST subset):\n")
    print(report)
    (OUT_DIR / "classification_report_test.txt").write_text(report, encoding="utf-8")

    # Creates classification report
    rep_dict = sk_classification_report(
        all_true, all_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    with (OUT_DIR / "per_class_metrics_test.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class","precision","recall","f1","num_test_instances"])
        for name in target_names:
            r = rep_dict.get(name, {})
            w.writerow([
                name,
                r.get("precision", 0.0),
                r.get("recall",    0.0),
                r.get("f1-score",  0.0),
                int(r.get("support", 0))
            ])

    acc = (sum(int(a == b) for a, b in zip(all_true, all_pred)) / max(1, len(all_true)))
    summary = {
        "subset_classes_evaluated": len(target_names),
        "images_evaluated": len(all_true),
        "images_skipped_nonoverlap": skipped_nonoverlap,
        "subset_accuracy": round(acc, 6),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"Skipped images (classes not in training): {skipped_nonoverlap}")
    log(f"Subset accuracy over {len(target_names)} classes: {acc:.4f}")
    log(f"Outputs saved in: {OUT_DIR}")
    log(f"Done in {summary['elapsed_sec']:.1f}s")

if __name__ == "__main__":
    main()
