import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from models.linear import SoftmaxLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters for search
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VALIDATION_SIZE = 5000  # Number of images to use for validation

transform = transforms.Compose([
    transforms.ToTensor()
])

# Load training dataset
full_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)

# Split training set into train and validation
indices = list(range(len(full_trainset)))
np.random.shuffle(indices)
train_idx, val_idx = indices[VALIDATION_SIZE:], indices[:VALIDATION_SIZE]

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
    model = SoftmaxLinear().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    # training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
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
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, model

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
