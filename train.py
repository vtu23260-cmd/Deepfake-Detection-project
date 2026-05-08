import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, random_split

# =====================
# CONFIG
# =====================
DATASET_PATH = "dataset/"
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001
VAL_SPLIT = 0.2
PATIENCE = 3

# =====================
# START
# =====================
print("🔥 Script started...")

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥 Using device:", device)

# =====================
# TRANSFORMS
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor()
])

# =====================
# LOAD DATASET
# =====================
print("📂 Loading dataset...")

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

print(f"✅ Total images found: {len(dataset)}")

# Split dataset
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📊 Train size: {train_size}, Val size: {val_size}")

# =====================
# LOAD MODEL
# =====================
print("📦 Loading EfficientNet model... (first time may take 1–2 min)")

try:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    print("✅ Pretrained weights loaded!")
except:
    print("⚠️ Could not download weights, using random init")
    model = efficientnet_b0(weights=None)

# Modify classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

print("🚀 Model ready!")

# =====================
# LOSS & OPTIMIZER
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =====================
# TRAINING LOOP
# =====================
best_val_loss = float("inf")
patience_counter = 0

print("🏁 Starting training...\n")

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    # TRAIN
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Print every 50 batches
        if i % 50 == 0:
            print(f"Batch {i}/{len(train_loader)} running...")

    train_acc = 100 * correct / total

    # VALIDATION
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(f"""
📊 Results:
Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%
Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%
""")

    # =====================
    # EARLY STOPPING
    # =====================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved!")

    else:
        patience_counter += 1
        print(f"⚠️ No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered!")
            break

print("\n🎉 Training complete!")