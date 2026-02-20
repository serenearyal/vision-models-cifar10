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
from torch.utils.data.sampler import SubsetRandomSampler
from models.cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Hyperparameters for search
NUM_EPOCHS = 10
BATCH_SIZE = 128
# 0.001 performed better (6% higher accuracy) than both 0.0001 and 0.01
LEARNING_RATE = 0.01

VALIDATION_SIZE = 5000  # Number of images to use for validation
RUN_HPARAM_SEARCH = False


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

# Load training dataset
full_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)

# Split training set into train and validation
indices = list(range(len(full_trainset)))       #[0, 1, 2, ..., 49999]
np.random.shuffle(indices)         #[35000, 35001, ..., 49999, 0,]


train_idx, val_idx = indices[VALIDATION_SIZE:], indices[:VALIDATION_SIZE]
#[last 45k elements] and [first 5k elements]


#sampler
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

trainloader = torch.utils.data.DataLoader(
    full_trainset, batch_size=BATCH_SIZE, sampler=train_sampler
)

valloader = torch.utils.data.DataLoader(
    full_trainset, batch_size=BATCH_SIZE, sampler=val_sampler
)

# Load test dataset
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False
)

def train_and_eval(weight_decay, train_loader, eval_loader, epochs=NUM_EPOCHS, silent=True):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # we can use different optimizers here, but SGD + momentum works best in this case
    # SGD + momentum > Adam > SGD
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay, momentum= 0.9)

    # training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # cross entropy loss (softmax + negative log likelihood loss)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if not silent:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

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

if RUN_HPARAM_SEARCH:
    # Hyperparameter search
    weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    best_wd = None
    best_val_acc = -1

    print(f"Starting hyperparameter search on validation set (Epochs: {NUM_EPOCHS})...")
    for wd in weight_decays:
        print(f"Testing weight decay: {wd}")
        val_acc, _ = train_and_eval(wd, trainloader, valloader)
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wd = wd

    print(f"\nBest weight decay found: {best_wd} (Val Accuracy: {best_val_acc:.2f}%)")

    # Final evaluation on test set
    print(f"\nPerforming final evaluation on test set with best weight decay ({best_wd})...")
    test_acc, final_model = train_and_eval(best_wd, trainloader, testloader, epochs=NUM_EPOCHS, silent=False)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
else:
    # Final evaluation on test set
    #ran the hyperparam search; best weight decay to be 0 (no regularization)
    # 1e-5 is close second best, almost identical accuracy
    best_wd = 0.00001
    print(f"\nPerforming final evaluation on test set with best weight decay ({best_wd})...")
    test_acc, final_model = train_and_eval(best_wd, trainloader, testloader, epochs=NUM_EPOCHS, silent=False)
    print(f"Final Test Accuracy: {test_acc:.2f}%")