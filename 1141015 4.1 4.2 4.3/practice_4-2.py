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
        self.dropout = nn.Dropout(0.25)
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

def train_validate(model, device, train_loader, val_loader, criterion, optimizer, epoch):
    # Training phase
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Validation Accuracy: {val_accuracy:.2f}%")
    return avg_train_loss, avg_val_loss, val_accuracy

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
patience = 3  # Stop if no improvement for 3 consecutive epochs
best_val_acc = 0.0
epochs_no_improve = 0
train_losses = []
val_accuracies = []
best_model_path = './cifar_net_best.pth'

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_losses, val_losses, val_accuracies = [], [], []
best_val_acc = 0

# Define scheduler (Whenever the verification accuracy does not improve for 2 consecutive times→ learning rate automatically x 0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)    

for epoch in range(1, epochs + 1):
    train_loss, val_loss, val_acc = train_validate(model, device, train_loader, val_loader, criterion, optimizer, epoch)
    scheduler.step(val_acc)  # Adjust learning rate based on val_acc
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Check for improvement
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        # torch.save(model.state_dict(), best_model_path)
        print(f"Validation accuracy improved to {best_val_acc:.2f}%. ")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    # Early stopping
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

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
