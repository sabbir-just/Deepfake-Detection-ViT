# =====================================================
# Optimized Deepfake Detection Training (Using Preprocessed Frames)
# =====================================================

import os, torch, timm, shutil, random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from torch.cuda.amp import autocast, GradScaler

# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type=="cuda": print("GPU:", torch.cuda.get_device_name(0))

# ------------------- Paths -------------------
WORK_PATH = "/kaggle/working/frames"  # preprocessed frames folder
MODEL_PATH = "/kaggle/working/vit_b16_best.pth"
CHECKPOINT_PATH = "/kaggle/working/vit_b16_ckpt.pth"

# ------------------- Dataset -------------------
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples, self.labels = [], []
        self.transform = transform
        classes = {"real":0, "fake":1}
        for cls in classes:
            cls_folder = "Celeb-real" if cls=="real" else "Celeb-synthesis"
            path = os.path.join(root_dir, cls_folder)
            if not os.path.exists(path): continue
            for img_file in os.listdir(path):
                if img_file.lower().endswith((".jpg",".png")):
                    self.samples.append(os.path.join(path,img_file))
                    self.labels.append(classes[cls])
        if len(self.samples)==0:
            raise ValueError(f"No images found in {root_dir}.")
        print("Total images loaded:", len(self.samples))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform: img = self.transform(img)
        return img, label

# ------------------- Transforms -------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1,0.1,0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

dataset = DeepfakeDataset(WORK_PATH)
train_size = int(0.8*len(dataset))
val_size = len(dataset)-train_size
train_dataset, val_dataset = random_split(dataset,[train_size,val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

labels = torch.tensor(dataset.labels)
class_counts = torch.bincount(labels)
class_weights = 1.0 / class_counts.float()
print("Class counts:", class_counts.tolist())
print("Class weights:", class_weights)

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=2,pin_memory=True)
val_loader   = DataLoader(val_dataset,batch_size=16,shuffle=False,num_workers=2,pin_memory=True)

# ------------------- Model -------------------
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)
scaler = GradScaler()

start_epoch, best_acc = 0, 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_acc"]
    print("Resuming from epoch", start_epoch)

EPOCHS = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ------------------- Training -------------------
for epoch in range(start_epoch, EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---------------- TRAIN ----------------
    model.train()
    running_loss = 0
    correct,total = 0,0

    for images, labels_batch in tqdm(train_loader):

        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        _, preds = torch.max(outputs,1)
        correct += (preds==labels_batch).sum().item()
        total += labels_batch.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    scheduler.step()

    # ---------------- VALIDATION ----------------
    model.eval()

    val_loss_total = 0
    val_correct,val_total = 0,0

    with torch.no_grad():
        for images, labels_batch in val_loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels_batch)

            val_loss_total += loss.item()

            _, preds = torch.max(outputs,1)

            val_correct += (preds==labels_batch).sum().item()
            val_total += labels_batch.size(0)

    val_loss = val_loss_total / len(val_loader)
    val_acc = val_correct / val_total

    # store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ----- Validation -----
    model.eval()
    val_correct,val_total = 0,0
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs,1)
            val_correct += (preds==labels_batch).sum().item()
            val_total += labels_batch.size(0)
    val_acc = val_correct/val_total
    print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # ----- Save best model -----
    if val_acc>best_acc:
        best_acc=val_acc
        torch.save(model.cpu().state_dict(), MODEL_PATH)
        model.to(device)
        print("Best model saved!")

    # ----- Save checkpoint -----
    torch.save({
        "epoch": epoch+1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc
    }, CHECKPOINT_PATH)
    print("Checkpoint saved!")

print("Training completed!")

# ------------------- Download Model (Kaggle) -------------------
from IPython.display import FileLink
print("Download best model:")
FileLink(MODEL_PATH)