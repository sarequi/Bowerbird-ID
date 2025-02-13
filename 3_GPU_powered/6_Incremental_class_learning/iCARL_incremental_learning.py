import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report, confusion_matrix

# This script performs iCARL-like incremental class learning on the pre trained  
# model (originally trained to recognise 16 birds). It:
#   1) Loads the old model
#   2) Load new data (new classes)
#   3) Expand the model's classifier outputs to include the new classes
#   4) Uses herding to select exemplar images from the old classes and combines them with new data
#   5) Train with cross-entropy + distillation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters for incremental learning
BATCH_SIZE = 32
NUM_WORKERS = 8
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
STEP_SIZE = 7
GAMMA = 0.1
MOMENTUM = 0.9
LR = 0.001
NUM_EPOCHS = 20
DISTILL_TEMPERATURE = 2.0
DISTILL_WEIGHT = 1.0
TOTAL_EXEMPLAR_BUDGET = 2000
IMAGE_SIZE = 512
HORIZONTAL_FLIP_PROB = 0.5

# Data transforms for train/val of new classes
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
    "val": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}

#  iCaRLModel: stores a feature extractor, a classifier, and old model copy

class ICaRLModel(nn.Module):
    def __init__(self, feature_extractor, feature_dim, initial_num_classes):
        super().__init__()
        # Feature extractor: typically the ResNet backbone without final FC
        self.feature_extractor = feature_extractor
        # Linear classifier, which can be expanded to more classes later
        self.classifier = nn.Linear(feature_dim, initial_num_classes, bias=True)
        # Old model for distillation
        self.old_model = None
        # Exemplar sets: dict mapping class idx -> list of (img_tensor, label)
        self.exemplar_sets = {}

    def forward(self, x):
        # Forward pass: extract features, flatten, then apply classifier
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

    def extract_features(self, x):
        # Extract features only, without final classifier
        feats = self.feature_extractor(x)
        return feats.view(feats.size(0), -1)

    def expand_classifier(self, new_total_classes):
        # Expand FC layer to have new_total_classes outputs
        old_classes = self.classifier.out_features
        if new_total_classes <= old_classes:
            return
        old_w = self.classifier.weight.data
        old_b = self.classifier.bias.data
        # Create new layer, copy old weights
        new_fc = nn.Linear(self.classifier.in_features, new_total_classes, bias=True)
        with torch.no_grad():
            new_fc.weight[:old_classes] = old_w
            new_fc.bias[:old_classes] = old_b
        self.classifier = new_fc

    def update_old_model(self):
        # Keep a frozen copy of the model for next distillation
        self.old_model = copy.deepcopy(self)
        for p in self.old_model.parameters():
            p.requires_grad = False

#  Herding: select exemplars from a dataset by approximating its mean feat

def herding_selection(feature_fn, dataset, n_exemplars):
    # feature_fn extracts features for each sample
    feature_fn.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    feats_all, imgs_all, labels_all = [], [], []
    
    with torch.no_grad():
        for imgs, lbs in loader:
            imgs = imgs.to(device)
            feats = feature_fn(imgs).cpu()
            feats_all.append(feats)
            imgs_all.append(imgs.cpu())
            labels_all.append(lbs)
    feats_all = torch.cat(feats_all, dim=0)
    imgs_all = torch.cat(imgs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    # Compute mean feature (L2-normalized)
    mean_feat = feats_all.mean(dim=0, keepdim=True)
    mean_feat = mean_feat / mean_feat.norm()
    selected_indices = []
    sum_exemplar = torch.zeros_like(mean_feat)

    # Iteratively pick samples that best approximate the mean
    for _ in range(n_exemplars):
        min_dist = 1e10
        chosen = -1
        for i in range(feats_all.size(0)):
            if i in selected_indices:
                continue
            candidate_sum = sum_exemplar + feats_all[i : i+1]
            candidate_mean = candidate_sum / candidate_sum.norm()
            dist = torch.norm(mean_feat - candidate_mean)
            if dist < min_dist:
                min_dist = dist
                chosen = i
        selected_indices.append(chosen)
        sum_exemplar += feats_all[chosen : chosen+1]

    # Build the list of exemplars
    ex_data = []
    for idx in selected_indices:
        ex_data.append((imgs_all[idx], labels_all[idx].item()))
    return ex_data

#  Reduce exemplar sets to maintain total_exemplar_budget across all classes

def reduce_exemplar_sets(icarl_model, budget):
    num_classes = len(icarl_model.exemplar_sets)
    if num_classes == 0:
        return
    # M = budget // number_of_classes
    m = budget // num_classes
    new_sets = {}
    for cid, ex_list in icarl_model.exemplar_sets.items():
        new_sets[cid] = ex_list[:m]  # keep only up to m
    icarl_model.exemplar_sets = new_sets

#  Distillation loss for preserving old model's knowledge (iCaRL-like)

def icarl_distillation_loss(new_logits, old_logits, T=DISTILL_TEMPERATURE):
    old_sm = F.softmax(old_logits / T, dim=1)
    new_logsm = F.log_softmax(new_logits / T, dim=1)
    return F.kl_div(new_logsm, old_sm, reduction="batchmean") * (T**2)

#  Main incremental training routine (iCaRL): add new classes to old model

def incremental_train_icarl(icarl_model, train_dataset, val_dataset, n_new_classes, epochs=NUM_EPOCHS, lr=LR):
    old_num = icarl_model.classifier.out_features
    new_total = old_num + n_new_classes
    icarl_model.expand_classifier(new_total) # expand outputs
    icarl_model.to(device)

    # We gather all samples in 'train_dataset' (which should have only the new classes),
    # group them by class index offset = old_num
    loader_1 = DataLoader(train_dataset, batch_size=1, shuffle=False)
    offset = old_num
    class_samples = {}
    for c in range(offset, offset + n_new_classes):
        class_samples[c] = []

    # For each sample in the new dataset, assign the "real label" = offset + dataset_label
    for imgs, labs in loader_1:
        real_lbl = labs.item() + offset
        class_samples[real_lbl].append((imgs.squeeze(0), real_lbl))

    # For each new class use herding to pick exemplars, then store them
    for cidx in range(offset, offset + n_new_classes):
        if len(class_samples[cidx]) == 0:
            continue
        imgs_t, labs_t = [], []
        for (im_, lb_) in class_samples[cidx]:
            imgs_t.append(im_)
            labs_t.append(lb_)
        imgs_t = torch.stack(imgs_t, dim=0)
        labs_t = torch.tensor(labs_t, dtype=torch.long)
        dset_t = TensorDataset(imgs_t, labs_t)
        ex_data = herding_selection(lambda x: icarl_model.extract_features(x.to(device)), dset_t, 2000)
        icarl_model.exemplar_sets[cidx] = ex_data

    # Rebalance exemplars so total <= TOTAL_EXEMPLAR_BUDGET
    reduce_exemplar_sets(icarl_model, TOTAL_EXEMPLAR_BUDGET)

    # Combine old exemplars + new training data
    old_exemplar_items = []
    for cid, exlist in icarl_model.exemplar_sets.items():
        old_exemplar_items.extend(exlist)
    if len(old_exemplar_items) > 0:
        ex_imgs, ex_lbls = [], []
        for (im_, lb_) in old_exemplar_items:
            ex_imgs.append(im_)
            ex_lbls.append(lb_)
        ex_imgs = torch.stack(ex_imgs, 0)
        ex_lbls = torch.tensor(ex_lbls, dtype=torch.long)
        ex_set = TensorDataset(ex_imgs, ex_lbls)
        # Combine new train set + old exemplars
        full_trainset = ConcatDataset([train_dataset, ex_set])
    else:
        # If no old exemplars exist, just use new train data
        full_trainset = train_dataset

    combined_loader = DataLoader(full_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader_ = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Set up training
    opt = optim.SGD(icarl_model.parameters(), lr=lr, momentum=MOMENTUM)
    sch = lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    best_wts = copy.deepcopy(icarl_model.state_dict())
    best_acc = 0.0

    # save the best model for this increment
    save_pth = os.path.join(os.getcwd(), f"best_increment_model_{new_total}_classes.pth")

    for epoch in range(epochs):
        icarl_model.train()
        run_loss = 0.0
        run_corr = 0
        total_s = 0
        for inps, labs in combined_loader:
            inps, labs = inps.to(device), labs.to(device)
            opt.zero_grad()
            outs = icarl_model(inps)
            ce_loss = nn.CrossEntropyLoss()(outs, labs)
            kd_loss = torch.tensor(0.0, device=device)
            if icarl_model.old_model is not None:
                with torch.no_grad():
                    old_outs = icarl_model.old_model(inps)
                    old_outs = old_outs[:, :old_num]
                kd_loss = icarl_distillation_loss(outs[:, :old_num], old_outs)
            loss = ce_loss + DISTILL_WEIGHT * kd_loss
            loss.backward()
            opt.step()
            run_loss += loss.item() * inps.size(0)
            _, prd = torch.max(outs, 1)
            run_corr += torch.sum(prd == labs).item()
            total_s += inps.size(0)

        # Validation 
        icarl_model.eval()
        val_corr = 0
        val_tot = 0
        with torch.no_grad():
            for vx, vy in val_loader_:
                vx, vy = vx.to(device), vy.to(device)
                vout = icarl_model(vx)
                _, vpred = torch.max(vout, 1)
                val_corr += torch.sum(vpred == vy).item()
                val_tot += vx.size(0)
        val_acc = val_corr / val_tot
        sch.step()
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(icarl_model.state_dict())
            torch.save(best_wts, save_pth)

    # Load best and freeze as old_model
    icarl_model.load_state_dict(best_wts)
    icarl_model.update_old_model()
    return icarl_model

#  Evaluate the model, printing classification report + confusion matrix

def evaluate_model(icarl_model, loader, class_names):
    icarl_model.eval()
    preds_all, labs_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = icarl_model(x)
            _, p = torch.max(out, 1)
            preds_all.extend(p.cpu().numpy())
            labs_all.extend(y.cpu().numpy())
    print(classification_report(labs_all, preds_all, target_names=class_names))
    cm = confusion_matrix(labs_all, preds_all)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(os.getcwd(), "confusion_matrix_incremental.png"))
    plt.close()

#  load new data

def load_data_for_increment(data_dir):
    train_d = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=data_transforms["train"])
    val_d = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=data_transforms["val"])
    return train_d, val_d, train_d.classes

#  MAIN: usage -> python incremental.py <old_model_path> <data_dir_new> <N> 
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python incremental.py <old_model_path> <data_dir_new> <num_new_classes>")
        sys.exit(1)
    old_model_path = sys.argv[1]
    data_dir_new = sys.argv[2]
    num_new_classes = int(sys.argv[3])
    train_ds_new, val_ds_new, new_class_names = load_data_for_increment(data_dir_new)

    # Load old model
    print(f"Loading old model from {old_model_path}")
    old_state = torch.load(old_model_path, map_location=device)
    base_resnet = models.resnet50(pretrained=False)
    feat_dim = base_resnet.fc.in_features
    feat_extractor = nn.Sequential(*list(base_resnet.children())[:-1])
    old_out_features = old_state["classifier.weight"].shape[0]
    icarl_model = ICaRLModel(feat_extractor, feat_dim, old_out_features).to(device)
    icarl_model.load_state_dict(old_state)

    # Perform incremental training with new classes
    icarl_model = incremental_train_icarl(
        icarl_model,
        train_ds_new,
        val_ds_new,
        num_new_classes,
        epochs=NUM_EPOCHS,
        lr=LR
    )

    # Evaluate on new validation set
    new_loader = DataLoader(val_ds_new, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    all_class_names = [f"OldClass{i}" for i in range(old_out_features)] + new_class_names
    evaluate_model(icarl_model, new_loader, all_class_names)
