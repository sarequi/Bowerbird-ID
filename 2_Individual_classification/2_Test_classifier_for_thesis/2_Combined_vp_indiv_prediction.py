import os
import csv
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torch import nn

VP_MODEL_PATH = "/lisc/data/scratch/becogbio/juarez/test_thesis/3_vp_classifier/trained_vp_clssfr/resnet18_viewpoint_best.pth"
VP_CLASS_DIR  = "/lisc/data/scratch/becogbio/juarez/test_thesis/3_vp_classifier/vp_data_for_training_vp_clssfr/train_val_test_data/train"

IND_MODEL_PATH = "/lisc/data/scratch/becogbio/juarez/test_thesis/2_individual_classifier/trained_individual_clssfr/all_viewpoints/training_outputs/best_model.pth"
IND_CLASS_DIR  = "/lisc/data/scratch/becogbio/juarez/test_thesis/2_individual_classifier/train_val_individual_clssfr/all_viewpoints/train"

IMAGE_DIR = "/lisc/data/scratch/becogbio/juarez/test_thesis/test_frames"
OUTPUT_CSV = "/lisc/data/scratch/becogbio/juarez/test_thesis/Combined_predict_vp_individual/combined_predictions.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
vp_classes = sorted(os.listdir(VP_CLASS_DIR))
vp_model = models.resnet18()
vp_model.fc = nn.Linear(vp_model.fc.in_features, len(vp_classes))
vp_model.load_state_dict(torch.load(VP_MODEL_PATH, map_location=DEVICE))
vp_model.to(DEVICE).eval()

ind_classes = sorted(os.listdir(IND_CLASS_DIR))
ind_model = models.resnet50()
ind_model.fc = nn.Linear(ind_model.fc.in_features, len(ind_classes))
ind_model.load_state_dict(torch.load(IND_MODEL_PATH, map_location=DEVICE))
ind_model.to(DEVICE).eval()

# Transformation
vp_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
ind_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

done_frames = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        next(f)  # skip header
        for line in f:
            done_frames.add(line.split(",")[0])  # first column = Frame Name

# Predicts and saves row by row
with open(OUTPUT_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if not done_frames:
        writer.writerow(["Frame Name", "Viewpoint_prediction", "Inddividual_prediction"])

    for bird_id in os.listdir(IMAGE_DIR):
        bird_path = os.path.join(IMAGE_DIR, bird_id)
        if not os.path.isdir(bird_path):
            continue
        for frame in tqdm(os.listdir(bird_path), desc=f"Predicting {bird_id}"):
            if frame in done_frames:
                continue
            frame_path = os.path.join(bird_path, frame)
            if not os.path.isfile(frame_path):
                continue

            try:
                img = Image.open(frame_path).convert("RGB")
                with torch.inference_mode():
                    vp_pred = vp_model(vp_transform(img).unsqueeze(0).to(DEVICE)).argmax(1).item()
                    ind_pred = ind_model(ind_transform(img).unsqueeze(0).to(DEVICE)).argmax(1).item()
                writer.writerow([frame, vp_classes[vp_pred], ind_classes[ind_pred]])
                f.flush()
            except Exception as e:
                print(f"[error] {frame_path}: {e}")

print(f"Predictions saved to {OUTPUT_CSV}")
