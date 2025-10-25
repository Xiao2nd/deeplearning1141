import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
torch.manual_seed(16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# Data Transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     # Read image ranged from [0, 1] and transform to [-1, 1], x-mean/std
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     # ImageNet
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


batch_size = 1000

train_ds = torchvision.datasets.STL10(root='./STL-10', split='train', download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = torchvision.datasets.STL10(root= './STL-10', split='test', download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print(train_ds.data.shape, test_ds.data.shape)


classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
import matplotlib.pyplot as plt
import numpy as np

# Image display function
def imshow(img):
    img = img * 0.5 + 0.5  # Restore the image
    npimg = img.numpy()
    # The color is shifted to the last dimension 
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # (a, b, c)  (b, c, a)
    plt.axis('off')
    plt.show()

# Take a batch of data
batch_size_tmp = 8
train_loader_tmp = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_tmp)
dataiter = iter(train_loader_tmp)
images, labels = next(dataiter)
print(images.shape)

# Display image
plt.figure(figsize=(10,6))
imshow(torchvision.utils.make_grid(images))
# Displays classes
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_tmp)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #(in_channel, out_channel, kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 21 * 21, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
#加上dropout準確率能再提升
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from torchinfo import summary
model = Net().to(device)
print(summary(model, (1, 3, 96, 96)))


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    loss_list = []    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #  if (batch_idx+1) % 10 == 0:
        # 記錄 batch 的 loss
        loss_list.append(loss.item())
        
        # 每個 batch 都印出來（因為 batch 數量最多5 永遠不會%==0）
        batch = (batch_idx+1) * len(data)
        data_count = len(train_loader.dataset)
        percentage = (100. * (batch_idx+1) / len(train_loader))
        print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ' +
              f'({percentage:.0f} %)  Loss: {loss.item():.6f}')
    return loss_list


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    # Average loss
    test_loss /= len(test_loader.dataset) 
    # Displays the test results
    data_count = len(test_loader.dataset)
    percentage = 100. * correct / data_count 
    print(f'Accuracy: {correct}/{data_count} ({percentage:.2f}%)')

epochs = 10
lr=0.1

# Build a model
model = Net().to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()
# Set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

loss_list = []
for epoch in range(1, epochs + 1):
    train_loss = train(model, device, train_loader, criterion, optimizer, epoch)
    loss_list += train_loss
    test(model, device, test_loader)


# Plot the loss of the training process
import matplotlib.pyplot as plt

plt.plot(loss_list, 'r')

PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)

model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)
print(model)

test(model, device, test_loader)

batch_size=8
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Display the image
plt.figure(figsize=(10,6))
imshow(torchvision.utils.make_grid(images))

print('Real class: ', ' '.join(f'{classes[labels[j]]:5s}' 
                         for j in range(batch_size)))

# predict
outputs = model(images.to(device))

_, predicted = torch.max(outputs, 1)

print('Predict class: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(batch_size)))

# Initialize the correct number for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# predict
batch_size=1000
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predictions = torch.max(outputs, 1)
        # Calculate the correct number for each class
        for label, prediction in zip(target, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# Calculate the accuracy of each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'{classname:5s}: {accuracy:.1f} %')
