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
                        num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Trains a ResNet50 model for a single subset size and returns 
    validation loss and accuracy across epochs
    Also saves training history (per epoch metrics) for plotting
    """

    DATA_DIR = os.path.join(data_base_dir, str(subset_size))
    OUTPUT_DIR = os.path.join(output_base_dir, str(subset_size))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    classification_report_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    cm_figure_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    training_history_path = os.path.join(OUTPUT_DIR, "training_history.csv")

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

    # creates datasets and dataloaders
    datasets_ = {
        phase: datasets.ImageFolder(os.path.join(DATA_DIR, phase), 
                                    transform=data_transforms[phase])
        for phase in ['train', 'val']
    }
    dataloaders = {
        phase: DataLoader(datasets_[phase], batch_size=batch_size,
                          shuffle=(phase == 'train'), num_workers=8)
        for phase in ['train', 'val']
    }
    dataset_sizes = {phase: len(datasets_[phase]) for phase in ['train', 'val']}
    class_names = datasets_['train'].classes

    print(f"Classes: {class_names}")
    print(f"Training samples: {dataset_sizes['train']}, Validation samples: {dataset_sizes['val']}")

    # loads classifier
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    best_model_wts = model.state_dict()
    
    val_losses, val_accuracies = [], []
    
    training_history = []  # stores a dictionary per epoch

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # tracks metrics in each epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': None, 'train_acc': None,
            'val_loss': None,   'val_acc': None
        }
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
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

            # updates scheduler after training
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # stores train/val metrics in the epoch_metrics dictionary
            if phase == 'train':
                epoch_metrics['train_loss'] = epoch_loss
                epoch_metrics['train_acc'] = epoch_acc.item()
            else:
                epoch_metrics['val_loss'] = epoch_loss
                epoch_metrics['val_acc'] = epoch_acc.item()

                # keeps track of the best validation accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

        # appends the metrics for plotting at the end of each epoch
        training_history.append(epoch_metrics)
        
        val_losses.append(epoch_metrics['val_loss'])
        val_accuracies.append(epoch_metrics['val_acc'])

    # load best weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, model_save_path)

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

    # saves classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(classification_report_path)

    # saves confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.savefig(cm_figure_path)
    plt.close()

    # saves epoch-by-epoch training history to CSV
    df_history = pd.DataFrame(training_history)
    df_history.to_csv(training_history_path, index=False)

    print(f"Training complete for subset {subset_size}. Best accuracy: {best_acc:.4f}")

    # returns final epoch-by-epoch validation metrics as before
    return subset_size, val_losses, val_accuracies

def main():
    subset_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
    data_base_dir = "/gpfs/data/fs72607/juarezs98/subsets_finetune/"
    output_base_dir = "/gpfs/data/fs72607/juarezs98/subsets_finetune/Outputs_finetune/"

    num_epochs = 10
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