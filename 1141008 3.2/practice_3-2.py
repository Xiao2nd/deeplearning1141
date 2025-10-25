import os
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

PATH_DATASETS = "" 
BATCH_SIZE = 1024  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# FashionMNIST
training_data = FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# labels_map = {
#     0: "T-shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# print(len(training_data)) # 60000

# print dataset graph
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
# print(img.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.LeakyReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.ReLU(x)
        output = x
        return output
    
from torchinfo import summary
model = Net()
print(summary(model, (1, 28, 28)))


epochs = 10
lr=0.0052

# Create DataLoader
train_loader = DataLoader(training_data, batch_size=600)

# Set optimizer
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

model.train() # set the model to training mode
loss_list = [] 


for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss_list.append(loss.item())
            batch = batch_idx * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * batch_idx / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' + f'  Loss: {loss.item():.6f}')

test_loader = DataLoader(test_data, shuffle=False, batch_size=test_data.targets.shape[0])

model.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)

    # sum up batch loss
    test_loss += criterion(output, target).item()
    pred = output.argmax(dim=1)  
    correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
batch = batch_idx * len(data)
data_count = len(test_loader.dataset)
percentage = 100. * correct / data_count
print(f'Average_loss: {test_loss:.4f}, correct_rate: {correct}/{data_count}' + f' ({percentage:.0f}%)\n')

predictions = []
actual_labels = test_data.targets[:10000].cpu().numpy()

with torch.no_grad():
    for i in range(10000):
        data, target = test_data[i][0], test_data[i][1]
        data = data.unsqueeze(0).to(device)
        # add batch dimension
        output = model(data)
        pred = output.argmax(dim=1).item()
        predictions.append(pred)

#convert predictions to numpy array for use confusion matrix
predictions = np.array(predictions)

#compute confusion matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(actual_labels, predictions))
