import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

def train_single_subset(subset_size, data_base_dir, output_base_dir,
                        num_epochs=30, batch_size=32, learning_rate=0.001):
    """
    Trains a ResNet50 model for a single subset size and returns
    validation loss and accuracy across epochs.
    Also saves:
      - best model weights
      - classification report (CSV)
      - confusion matrix (PNG)
      - training curves (PNG) with loss and accuracy for train/val
    """

    DATA_DIR = os.path.join(data_base_dir, str(subset_size))
    OUTPUT_DIR = os.path.join(output_base_dir, str(subset_size))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    classification_report_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    cm_figure_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    training_curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data transformations
    IMAGE_SIZE = 512
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
    }

    # datasets and dataloaders
    datasets_ = {
        phase: datasets.ImageFolder(os.path.join(DATA_DIR, phase),
                                    transform=data_transforms[phase])
        for phase in ['train', 'val']
    }
    dataloaders = {
        phase: DataLoader(datasets_[phase], batch_size=batch_size,
                          shuffle=(phase == 'train'), num_workers=8, pin_memory=True)
        for phase in ['train', 'val']
    }
    dataset_sizes = {phase: len(datasets_[phase]) for phase in ['train', 'val']}
    class_names = datasets_['train'].classes

    print(f"Classes: {class_names}")
    print(f"Training samples: {dataset_sizes['train']}, Validation samples: {dataset_sizes['val']}")

    # model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    best_model_wts = model.state_dict()

    # history containers for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
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

            epoch_loss = running_loss / max(1, dataset_sizes[phase])
            epoch_acc = (running_corrects.double() / max(1, dataset_sizes[phase])).item()

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

    # save best weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, model_save_path)
    print(f"Best val Acc: {best_acc:.4f}")

    # evaluate final model on the validation set for classification report
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # classification report (CSV)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(classification_report_path)

    # confusion matrix (PNG)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_figure_path)
    plt.close()

    # training curves (PNG): Loss and Accuracy over epochs (train vs val)
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(training_curves_path)
    plt.close()

    print(f"Training curves saved to: {training_curves_path}")
    print(f"Training complete for subset {subset_size}. Best accuracy: {best_acc:.4f}")

    # return as before
    return subset_size, val_losses, val_accuracies

def main():
    subset_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
    data_base_dir = "/lisc/data/scratch/becogbio/juarez/thesis/4.2_Baseline_subsets/train_val_data/"
    output_base_dir = "/lisc/data/scratch/becogbio/juarez/thesis/4.2_Baseline_subsets/training_outputs/"

    num_epochs = 25
    batch_size = 32
    learning_rate = 0.001

    training_results = {}

    for subset_size in subset_sizes:
        print("=" * 50)
        print(f"Starting training for subset size: {subset_size}")
        print("=" * 50)

        subset, val_losses, val_accuracies = train_single_subset(
            subset_size=subset_size,
            data_base_dir=data_base_dir,
            output_base_dir=output_base_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        training_results[subset] = {
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }

if __name__ == "__main__":
    main()
