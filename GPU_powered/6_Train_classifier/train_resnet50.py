# Import required libraries
import os
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

# Detect GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
IMAGE_SIZE = 512
HORIZONTAL_FLIP_PROB = 0.5
BATCH_SIZE = 32
NUM_WORKERS = 8
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 16
USE_PRETRAINED = True
NUM_EPOCHS = 300
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1
DATA_DIR = '/gpfs/data/fs72607/juarezs98/train_val_data'
EVAL_BATCH_SIZE = 16

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}

# Load dataset
datasets = {
    phase: datasets.ImageFolder(os.path.join(DATA_DIR, phase), transform=data_transforms[phase])
    for phase in ['train', 'val']
}

dataloaders = {
    phase: DataLoader(datasets[phase], batch_size=BATCH_SIZE, shuffle=(phase == 'train'), num_workers=NUM_WORKERS)
    for phase in ['train', 'val']
}

dataset_sizes = {phase: len(datasets[phase]) for phase in ['train', 'val']}
class_names = datasets['train'].classes

print(f"Classes: {class_names}")
print(f"Training samples: {dataset_sizes['train']}, Validation samples: {dataset_sizes['val']}")

# Load pre-trained ResNet50
model = models.resnet50(pretrained=USE_PRETRAINED)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    val_losses = []
    val_accuracies = []

    model_save_path = os.path.join(os.getcwd(), "best_model.pth")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, model_save_path)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    print(f'Best model saved at: {model_save_path}')

    model.load_state_dict(best_model_wts)

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

# Train the model
model = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

# Evaluate the model
val_loader = dataloaders['val']
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Classification report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predictions")
plt.ylabel("Ground truth")
plt.title("Confusion matrix")
plt.show()