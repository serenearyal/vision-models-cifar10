import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from tqdm import tqdm

# Seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Data
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)

# model
model = vit_b_16(weights="IMAGENET1K_V1")

# Freeze everything (encoder, conv_proj, class_token, pos embeddings, etc.)


FREEZE_BACKBONE = True # set to False for full fine-tuning
if FREEZE_BACKBONE:
    for param in model.parameters():
        param.requires_grad = False

# Replace classification head (new layer gets requires_grad=True automatically)
model.heads.head = nn.Linear(model.heads.head.in_features, 10)

model = model.to(device)

# Optimizer (only trains parameters with requires_grad=True)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,
    weight_decay=0.05,
)
criterion = nn.CrossEntropyLoss()

# Training
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(trainloader)
    epoch_acc = evaluate(model, testloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Test Acc: {epoch_acc:.2f}%")

acc = evaluate(model, testloader)
print(f"\n✅ DONE!\nFinal Test Accuracy: {acc:.2f}%")