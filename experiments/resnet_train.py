import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.resnet import SimpleResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-4

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

# Load training dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform_train
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True
)

# Load test dataset
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform_test
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False
)

def train_and_eval(train_loader, eval_loader, epochs=NUM_EPOCHS, silent=False):
    model = SimpleResNet().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        if not silent:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # evaluation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, model

# Train and evaluate on test set
print(f"Training ResNet on CIFAR-10 (Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}, WD: {WEIGHT_DECAY})...")
test_acc, final_model = train_and_eval(trainloader, testloader, epochs=NUM_EPOCHS)
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
