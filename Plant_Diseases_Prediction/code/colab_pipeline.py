# Cell 1: Mount Drive & Define Architecture
import os
import torch
import torch.nn as nn
import torchvision.models as models
from google.colab import drive

# 1. Mount Google Drive for persistent storage
drive.mount('/content/drive')
print("✓ Google Drive mounted.")

# --- ARCHITECTURE DEFINITION ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class MobileNetV3_CBAM(nn.Module):
    def __init__(self, num_classes=28, pretrained=True):
        super(MobileNetV3_CBAM, self).__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_large(weights=weights)

        self.features = base_model.features
        self.cbam = CBAM(in_channels=960)
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier

        in_features = self.classifier[3].in_features
        self.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

print("✓ MobileNetV3_CBAM Architecture loaded into memory.")

# Cell 2: Phase 1 - PlantVillage Pre-training
import kagglehub
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Download PlantVillage
print("Downloading PlantVillage dataset...")
pv_path = kagglehub.dataset_download("emmarex/plantdisease")
PLANTVILLAGE_DIR = os.path.join(pv_path, "PlantVillage")
if not os.path.exists(PLANTVILLAGE_DIR):
    PLANTVILLAGE_DIR = pv_path

# 2. Data Loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(PLANTVILLAGE_DIR, transform=transform)
NUM_PV_CLASSES = len(full_dataset.classes)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# 3. Initialize Model & Optimizer
model = MobileNetV3_CBAM(num_classes=NUM_PV_CLASSES, pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 4. Train Loop
num_epochs = 15
best_val_acc = 0.0

print("\nStarting Phase 1: Pre-training on PlantVillage...")
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_acc = correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    val_acc = val_correct / val_total
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = "/content/drive/MyDrive/mobilenet_cbam_plantvillage_base.pth"
        torch.save(model.state_dict(), save_path)

print(f"\n✓ Phase 1 Complete! Weights safely backed up to: {save_path}")

# Cell 3: Extract PlantDoc Data
import zipfile

zip_path = '/content/drive/MyDrive/Datasets/classification_dataset.zip'
extract_path = '/content/'

if os.path.exists(zip_path):
    print(f"Found dataset at {zip_path}. Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✓ Dataset successfully restored to /content/train and /content/val")
else:
    print(f"ERROR: Could not find {zip_path}. Check your Google Drive path.")

# Cell 4: Phase 2 - Unfrozen Fine-Tuning
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLANTDOC_DIR = "/content" # Extracts directly to root /content/train and /content/val

# Heavy augmentation for real-world field adaptation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(PLANTDOC_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(PLANTDOC_DIR, "val"), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
NUM_PLANTDOC_CLASSES = len(train_dataset.classes)

# 1. Load PlantVillage weights (Assuming 15 classes from Phase 1)
model = MobileNetV3_CBAM(num_classes=15, pretrained=False)
model.load_state_dict(torch.load("/content/drive/MyDrive/mobilenet_cbam_plantvillage_base.pth"))

# 2. Swap the head to 28 classes
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, NUM_PLANTDOC_CLASSES)
model = model.to(device)

# 3. Ensure the ENTIRE network is unfrozen
for param in model.parameters():
    param.requires_grad = True

# 4. Ultra-low learning rate
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

num_epochs = 20
best_val_acc = 0.0

print("\nStarting Phase 2 (Unfrozen): Full-Network Fine-Tuning on PlantDoc...")
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_acc = correct / total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    scheduler.step(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "/content/drive/MyDrive/mobilenetv3_cbam_plantdoc_unfrozen_best.pth")

print(f"\n✓ Unfrozen Phase 2 Complete! Peak Validation Accuracy: {best_val_acc*100:.2f}%")

# FINAL OPTIMIZED RUN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLANTDOC_DIR = "/content" # Pointing to the extracted train/val folders

# Heavy Augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), # Added vertical flip
    transforms.RandomRotation(30),   # Increased rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(PLANTDOC_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(PLANTDOC_DIR, "val"), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
NUM_PLANTDOC_CLASSES = len(train_dataset.classes)

# 1. Initialize with standard ImageNet weights (pretrained=True)
# This is the approach that previously got you to 67.93%
model = MobileNetV3_CBAM(num_classes=NUM_PLANTDOC_CLASSES, pretrained=True)

# Increase dropout in the classifier to heavily penalize overfitting
model.classifier[2].p = 0.5
model = model.to(device)

# 2. Advanced Regularization
# Label smoothing prevents the model from being 100% confident, forcing it to learn better general features
criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

# Higher weight decay (1e-3 instead of 1e-4) to penalize large weights
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

num_epochs = 30 # Increased epochs because regularization slows down learning
# Cosine Annealing smoothly drops the learning rate following a cosine curve
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

best_val_acc = 0.0

print(f"\nStarting Final Optimized Run: ImageNet Base + CBAM + Label Smoothing...")
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_acc = correct / total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    scheduler.step() # Step the cosine scheduler

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "/content/drive/MyDrive/mobilenetv3_cbam_plantdoc_FINAL.pth")

print(f"\n✓ Optimized Run Complete! Peak Validation Accuracy: {best_val_acc*100:.2f}%")

# ROC CURVE PLOTTING
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Load the absolute best weights
model.load_state_dict(torch.load("/content/drive/MyDrive/mobilenetv3_cbam_plantdoc_FINAL.pth"))
model.eval()

y_true = []
y_score = []
class_names = val_dataset.classes
n_classes = len(class_names)

print("Evaluating best model to generate ROC curves...")
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_score.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_score = np.array(y_score)

# Binarize the output for multi-class ROC computation
y_true_bin = label_binarize(y_true, classes=range(n_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(12, 10))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'pink'])

# Only plot the top 5 and bottom 5 performing classes to keep the graph readable
sorted_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
top_5_classes = [x[0] for x in sorted_auc[:5]]
bottom_5_classes = [x[0] for x in sorted_auc[-5:]]
classes_to_plot = top_5_classes + bottom_5_classes

for i, color in zip(classes_to_plot, colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) - MobileNetV3+CBAM', fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig('/content/roc_curve.png', dpi=300)
print("✓ ROC curve saved as 'roc_curve.png'")
plt.show()

# ENVIRONMENT RESTORE
import os
import torch
import torch.nn as nn
import torchvision.models as models
import zipfile
from google.colab import drive

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Extract Dataset
zip_path = '/content/drive/MyDrive/Datasets/classification_dataset.zip'
extract_path = '/content/'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✓ Dataset extracted.")
else:
    print(f"ERROR: Could not find {zip_path}")

print("✓ Environment completely restored!")

# System: MobileNetV3 + CBAM Optimization Pipeline
# Author: Govardhan T

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLANTDOC_DIR = "/content" # Extracts directly to root /content/train and /content/val

# --- 1. FOCAL LOSS IMPLEMENTATION ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        # PyTorch natively supports probability targets from MixUp/CutMix
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# --- 2. ADVANCED V2 AUGMENTATION ---
# Upgrading to Torchvision v2 transforms for native CutMix/MixUp support
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(30),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(PLANTDOC_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(PLANTDOC_DIR, "val"), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

NUM_PLANTDOC_CLASSES = len(train_dataset.classes)

# --- 3. MIXUP & CUTMIX INITIALIZATION ---
# These transforms expect batches of samples as input, not individual images
cutmix = v2.CutMix(num_classes=NUM_PLANTDOC_CLASSES)
mixup = v2.MixUp(num_classes=NUM_PLANTDOC_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

# --- 4. MODEL SETUP ---
model = MobileNetV3_CBAM(num_classes=NUM_PLANTDOC_CLASSES, pretrained=True)
model.classifier[2].p = 0.5 # Maintain heavy dropout
model = model.to(device)

criterion = FocalLoss(gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
num_epochs = 40 # Increased epochs because CutMix takes longer to converge

# Cosine Annealing with Warm Restarts prevents the optimizer from getting stuck in local minima
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

best_val_acc = 0.0

print(f"\nStarting Final Run: Focal Loss + CutMix + MixUp...")
for epoch in range(num_epochs):

    # --- TRAINING LOOP ---
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Apply CutMix or MixUp dynamically to the batched tensors
        images, labels = cutmix_or_mixup(images, labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)
    scheduler.step()

    # --- VALIDATION LOOP ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Convert validation integer labels to one-hot for Focal Loss compatibility
            labels_one_hot = F.one_hot(labels, num_classes=NUM_PLANTDOC_CLASSES).float()
            loss = criterion(outputs, labels_one_hot)

            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "/content/drive/MyDrive/mobilenetv3_cbam_plantdoc_MAX.pth")

print(f"\n✓ Optimization Complete! Peak Validation Accuracy: {best_val_acc*100:.2f}%")

from google.colab import files

# Path to your best-performing model
best_model_path = "/content/drive/MyDrive/mobilenetv3_cbam_plantdoc_FINAL.pth"

print(f"Initiating download for: {best_model_path}")
files.download(best_model_path)

# PREDICTION SCRIPT
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# --- 1. CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/content/drive/MyDrive/mobilenetv3_cbam_plantdoc_FINAL.pth"
TEST_IMAGE_PATH = "/content/images.jpg" # Update this to the name of your uploaded image

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy'
]

# --- 2. LOAD THE MODEL ---
print("Loading model...")
model = MobileNetV3_CBAM(num_classes=len(CLASS_NAMES), pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval() # Set to evaluation mode (turns off dropout)

# --- 3. DEFINE IMAGE TRANSFORMS ---
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. RUN INFERENCE ---
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = inference_transform(image).unsqueeze(0) # Add batch dimension
    input_tensor = input_tensor.to(device)

    print(f"Running inference on {image_path}...")
    with torch.no_grad():
        outputs = model(input_tensor)

        # Apply softmax to get percentages (0.0 to 1.0)
        probabilities = F.softmax(outputs[0], dim=0)

        # Get the top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)

    print("\n--- PREDICTION RESULTS ---")
    for i in range(3):
        idx = top3_indices[i].item()
        prob = top3_prob[i].item() * 100
        predicted_class = CLASS_NAMES[idx]
        print(f"{i+1}. {predicted_class}: {prob:.2f}% confidence")

# Execute the prediction
predict_image(TEST_IMAGE_PATH)

# System: YOLOv8-cls Optimization & Export Pipeline
# Author: Govardhan T

import os
import zipfile
from google.colab import drive
from ultralytics import YOLO

# 2. Mount Google Drive to access the dataset and save results
drive.mount('/content/drive')
print("✓ Google Drive mounted.")

# 3. Extract the PlantDoc dataset
zip_path = '/content/drive/MyDrive/Datasets/classification_dataset.zip'
extract_path = '/content/'

if os.path.exists(zip_path):
    print(f"Extracting dataset from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✓ Dataset successfully restored to /content/train and /content/val")
else:
    print(f"ERROR: Could not find {zip_path}. Please check your Drive path.")

# 4. Initialize the YOLOv8 Nano Classification Model
print("\nInitializing YOLOv8-nano classification model...")
model = YOLO('yolov8n-cls.pt')

# 5. Execute the Training Loop
print("\nStarting Training Phase...")
results = model.train(
    data='/content',               # Path containing the 'train' and 'val' folders
    epochs=40,                     # 40 epochs is a strong baseline for YOLO
    imgsz=224,                     # Standard image size for mobile deployment
    batch=32,                      # Batch size
    device=0,                      # Force GPU usage
    project='/content/drive/MyDrive/PlantDoc_YOLO', # Save directly to Drive
    name='optimized_run_v1',       # Run name
    pretrained=True                # Start with ImageNet weights
)

print(f"\n✓ Training Complete! Peak metrics saved to Google Drive.")

# 6. Automatic TFLite Export for Mobile Deployment
print("\nExporting model for offline mobile inference...")
export_path = model.export(format='tflite', int8=True)

print(f"\n✓ Mobile deployment file (.tflite) successfully generated at: {export_path}")

from google.colab import drive
from google.colab import files
import os

# 1. Re-mount your Google Drive
drive.mount('/content/drive')

# 2. Define the path to your newly generated INT8 TFLite model
tflite_path = '/content/drive/MyDrive/PlantDoc_YOLO/optimized_run_v1/weights/best_int8.tflite'

# 3. Download to your local machine
if os.path.exists(tflite_path):
    print(f"Found model! Downloading {tflite_path}...")
    files.download(tflite_path)
else:
    print("Error: Could not find the model. Please check your Google Drive folders manually.")
