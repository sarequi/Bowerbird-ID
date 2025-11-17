import os
import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torch import nn

IMAGE_SIZE = 224
MODEL_PATH = "resnet18_viewpoint_best.pth"
CLASS_DIR = "vp_data_for_training_vp_clssfr/train_val_test_data/train"
IMAGE_DIR = "test_thesis/test_frames"
OUTPUT_CSV = "3_vp_classifier/test_on_test_set/viewpoint_predictions.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = sorted(os.listdir(CLASS_DIR))
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(class_names))
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()] if pred.item() < len(class_names) else f"Class{pred.item()}"

entries = []
for bird_id in os.listdir(IMAGE_DIR):
    bird_path = os.path.join(IMAGE_DIR, bird_id)
    if not os.path.isdir(bird_path):
        continue
    for frame in os.listdir(bird_path):
        frame_path = os.path.join(bird_path, frame)
        if not os.path.isfile(frame_path):
            continue
        try:
            pred = predict(frame_path)
        except Exception as e:
            print(f"[error] Failed on {frame_path}: {e}")
            pred = "Error"
        entries.append({"Frame Name": frame, "Prediction": pred})

df = pd.DataFrame(entries)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to: {OUTPUT_CSV}")
