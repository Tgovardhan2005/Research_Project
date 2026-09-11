import os

# 1. Create a hidden kaggle folder and move the API key there
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 2. Download the verified PlantVillage dataset via Kaggle CMD
print("Downloading PlantVillage...")
!kaggle datasets download -d abdallahalidev/plantvillage-dataset

# 3. Unzip it silently (-q) into the Stage2_Dataset folder
print("Unzipping...")
# Note: The downloaded zip file name matches the Kaggle slug
!unzip -q plantvillage-dataset.zip -d /content/Stage2_Dataset

print("✅ Ready for training!")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# The Kaggle dataset usually extracts into a 'color' folder
data_dir = '/content/Stage2_Dataset/plantvillage dataset/color'

# Define Data Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the entire dataset
full_dataset = datasets.ImageFolder(data_dir)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"✅ Loaded {num_classes} disease classes from PlantVillage.")

# Split into 80% Train / 20% Validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply transforms
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

# Create the Loaders that were missing!
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# Initialize MobileNetV3-Small
weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

# Freeze early layers for faster training
for param in model.features.parameters():
    param.requires_grad = False

# Swap the final classification head to match our 38 classes
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
print("✅ Setup complete! Proceed to the training loop.")

import time
import copy
from tqdm.auto import tqdm # 1. Import the progress bar library

epochs = 15
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

print("🚀 Starting Stage 2 MobileNetV3 Training...")
start_time = time.time()

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        # 2. Wrap the dataloader in tqdm to create the visual progress bar
        pbar = tqdm(dataloader, desc=f"{phase.capitalize()} Phase", leave=False)

        for inputs, labels in pbar:
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

            # 3. Update the progress bar text with the live loss
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f"✅ {phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), '/content/best_stage2_mobilenet.pth')
            print("🌟 New best model saved!")

time_elapsed = time.time() - start_time
print(f"\n🎉 Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")

from google.colab import files

# Download the Stage 2 MobileNetV3 weights
print("Downloading Stage 2 weights...")
files.download('/content/best_stage2_mobilenet.pth')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from google.colab import files

print("📊 Generating Confusion Matrix for Stage 2...")

# 1. Gather all predictions and true labels from the validation set
y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# 2. Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# 3. Plot the matrix using Seaborn
plt.figure(figsize=(22, 20)) # Large size to fit all 38 classes clearly
sns.heatmap(cm, annot=False, cmap='Blues', fmt='d',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Disease', fontsize=16, fontweight='bold')
plt.ylabel('Actual Disease', fontsize=16, fontweight='bold')
plt.title('Stage 2: MobileNetV3 Diagnostician Confusion Matrix', fontsize=20, fontweight='bold')
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# 4. Save and Download
save_path = '/content/stage2_confusion_matrix.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print("✅ Downloading matrix to your computer...")
files.download(save_path)

import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

print("📸 Generating Sample Predictions for Stage 2...")

def imshow(inp, title=None):
    """Denormalize and display the image tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Undo the normalization we applied during training
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        # Highlight correct predictions in green, incorrect in red
        color = 'green' if 'True' in title and title.split('Pred: ')[1].split('\n')[0] == title.split('True: ')[1] else 'red'
        plt.title(title, fontsize=9, color=color, fontweight='bold')
    plt.axis('off')

# Grab a single batch of validation data
model.eval()
inputs, labels = next(iter(val_loader))
inputs = inputs.to(device)

# Make predictions
with torch.no_grad():
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

# Plot a 3x3 grid
fig = plt.figure(figsize=(12, 12))
for i in range(9):
    ax = fig.add_subplot(3, 3, i+1)
    pred_name = class_names[preds[i]]
    true_name = class_names[labels[i]]

    # Shorten names slightly if they are too long for the plot
    pred_name = pred_name.replace(" ", "\n") if len(pred_name) > 15 else pred_name
    true_name = true_name.replace(" ", "\n") if len(true_name) > 15 else true_name

    imshow(inputs.cpu().data[i], title=f'Pred: {pred_name}\nTrue: {true_name}')

plt.tight_layout()
save_path = '/content/stage2_sample_predictions.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print("✅ Downloading prediction grid to your computer...")
files.download(save_path)