import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True "
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the final layer to output 10 classes (cats and dogs)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10- categories classification task

 #Freeze the parameters of all layers except last layer
for name, param in model.named_parameters():
    if 'fc' not in name:  # Only train last layer fc
        param.requires_grad = False

print(model)


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load training and test datasets
# train_data = datasets.ImageFolder('1141022 4.4\dog_vs_cat/train', transform=transform)
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# test_data = datasets.ImageFolder('1141022 4.4\dog_vs_cat/test', transform=transform)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

#load CIFER-10 dataset
batch_size = 256  # Reduced from 1000 to 32 to fit in GPU memory

train_data = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = torchvision.datasets.CIFAR10(root= './CIFAR10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)




# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
model.to(device)  # Move model to device before training
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # Move data to the appropriate device (e.g., GPU if available)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        running_loss += loss.item()
        
        # Clear cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

print('Training complete')

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation for efficiency
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total} %')

#Freeze the parameters of all layers except last layer
# for name, param in model.named_parameters():
#     if 'fc' not in name:  # Only train last layer fc
#         param.requires_grad = False
