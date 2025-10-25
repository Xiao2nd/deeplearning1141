import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed
torch.manual_seed(16)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download STL-10 dataset
dataset = torchvision.datasets.STL10(root='./STL-10', split='train', download=True, transform=transform)

# Split into train (80%) and val (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])


# Data loaders
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
# Test set
test_ds = torchvision.datasets.STL10(root='./STL-10', split='test', download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
# Class names
classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
# CNN Model with dropout
class Net(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 21 * 21, 120)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout_p = dropout_p
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function
def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Validation function
def evaluate(model, device, loader, criterion):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100. * correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

# Training loop with dynamic dropout adjustment
epochs = 20
model = Net(dropout_p=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_losses, val_losses, val_accuracies = [], [], []
best_val_acc = 0

for epoch in range(1, epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, device, val_loader, criterion)

    # Adjust dropout
    if val_acc > best_val_acc:
        model.dropout_p = max(0.1, model.dropout_p - 0.05)  # Decrease dropout
        print(f"Val accuracy improved to {val_acc:.2f}% -> decreasing dropout to {model.dropout_p:.2f}")
        best_val_acc = val_acc
    else:
        model.dropout_p = min(0.7, model.dropout_p + 0.05)  # Increase dropout
        print(f"Val accuracy did not improve -> increasing dropout to {model.dropout_p:.2f}")
    model.dropout.p = model.dropout_p  # Update dropout layer

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

# Plot training and validation loss
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test accuracy
_, test_acc = evaluate(model, device, test_loader, criterion)
print(f"Test Accuracy: {test_acc:.2f}%")
