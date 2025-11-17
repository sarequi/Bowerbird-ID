from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DATA_DIR = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/vp_data_for_training_vp_clssfr/train_val_test_data/")
OUT_DIR  = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/trained_vp_clssfr/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 100
BATCH = 32
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transforms
tf_train = T.Compose([
    T.Resize(256), T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.RandomRotation(7), T.ColorJitter(0.15, 0.15, 0.15, 0.05),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])
tf_eval = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

CLASSES_TO_KEEP = ["back", "front", "left_side", "right_side"]

class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, classes_to_keep=None, **kwargs):
        self._classes_to_keep = set(classes_to_keep) if classes_to_keep else None
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str):
        candidates = [
            d.name for d in Path(directory).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if self._classes_to_keep is not None:
            candidates = [c for c in candidates if c in self._classes_to_keep]

        candidates.sort()
        if not candidates:
            raise FileNotFoundError(
                f"No valid class folders found under {directory} after filtering."
            )
        class_to_idx = {c: i for i, c in enumerate(candidates)}
        return candidates, class_to_idx

# --- Datasets & loaders: only train and validation ---
splits, loaders = {}, {}
for split in ("train", "validation"):
    ds = FilteredImageFolder(
        DATA_DIR / split,
        classes_to_keep=CLASSES_TO_KEEP,
        transform=tf_train if split == "train" else tf_eval,
    )
    splits[split] = ds
    loaders[split] = DataLoader(
        ds, batch_size=BATCH, shuffle=(split == "train"),
        num_workers=4, pin_memory=True
    )

# model, loss, optimiser
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(CLASSES_TO_KEEP))  # 4 classes
model.to(DEVICE)

loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.05)
optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

#  train / validate
def epoch_pass(train_mode: bool):
    model.train(train_mode)
    split = "train" if train_mode else "validation"
    total_loss = total_correct = total_items = 0
    for x, y in loaders[split]:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimiser.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = loss_fn(logits, y)
            if train_mode:
                loss.backward()
                optimiser.step()
        total_loss    += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_items   += y.size(0)
    return total_loss / total_items, total_correct / total_items

best_val_acc = 0.0
for _ in range(EPOCHS):
    tr_loss, tr_acc = epoch_pass(train_mode=True)
    vl_loss, vl_acc = epoch_pass(train_mode=False)
    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)
    scheduler.step()

    # keep best checkpoint by val acc (optional but handy)
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), OUT_DIR / "resnet18_viewpoint_best.pth")

# save curves 
np.save(OUT_DIR / "history.npy", history)
for metric in ("acc", "loss"):
    plt.figure()
    plt.plot(history[f"train_{metric}"], label="train")
    plt.plot(history[f"val_{metric}"],   label="val")
    plt.xlabel("epoch"); plt.ylabel(metric); plt.legend()
    plt.savefig(OUT_DIR / f"{metric}_curve.png", dpi=150)
    plt.close()

# "Final" evaluation on the validation split (since no test split)
model.eval(); y_true, y_pred = [], []
with torch.no_grad():
    for x, y in loaders["validation"]:
        y_true.extend(y.tolist())
        y_pred.extend(model(x.to(DEVICE)).argmax(1).cpu().tolist())

np.savetxt(OUT_DIR / "confusion_matrix_val.txt",
           confusion_matrix(y_true, y_pred), fmt="%d")

with open(OUT_DIR / "classification_report_val.txt", "w") as f:
    f.write(classification_report(y_true, y_pred,
                                  target_names=splits["validation"].classes, digits=4))

# also save the final weights from the last epoch
torch.save(model.state_dict(), OUT_DIR / "resnet18_viewpoint_last.pth")
