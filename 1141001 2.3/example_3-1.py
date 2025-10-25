import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = "" 
BATCH_SIZE = 1024  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download MNIST training data at the same directory with current program file
train_ds = MNIST(PATH_DATASETS, train=True, download=True, 
                 transform=transforms.ToTensor())

# download MNIST test data
test_ds = MNIST(PATH_DATASETS, train=False, download=True, 
                 transform=transforms.ToTensor())

# the shape of training data and test data
print(train_ds.data.shape, test_ds.data.shape)
print(train_ds.targets.shape, test_ds.targets.shape)


# print(train_ds.targets[:10])

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# import matplotlib.pyplot as plt

# X = train_ds.data[0] # without normalization

# plt.imshow(X.reshape(28,28), cmap='gray')

# plt.show() 


# Create model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28 * 28, 256), 
    torch.nn.Linear(256, 10), 
).to(device)

epochs = 5
lr=0.1

# Create DataLoader
train_loader = DataLoader(train_ds, batch_size=600)

# Set optimizer
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
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


import matplotlib.pyplot as plt
plt.plot(loss_list, 'r')

test_loader = DataLoader(test_ds, shuffle=False, batch_size=test_ds.targets.shape[0])

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
with torch.no_grad():
    for i in range(20):
        data, target = test_ds[i][0], test_ds[i][1]
        data = data.reshape(1, *data.shape).to(device) # *:Unpack the list into independent parameters
        output = torch.argmax(model(data), axis=-1)
        predictions.append(str(output.item()))

print('actual    :', test_ds.targets[0:20].numpy())
print('prediction: ', ' '.join(predictions[0:20]))

import numpy as np

i=8
data = test_ds[i][0]
data = data.reshape(1, *data.shape).to(device) # *:Unpack the list into independent parameters

#print(data.shape)
predictions = torch.softmax(model(data), dim=1)
print(f'0~9 predict rate: {np.around(predictions.cpu().detach().numpy(), 2)}')
print(f'0~9 predict rate: {np.argmax(predictions.cpu().detach().numpy(), axis=-1)}')


# save model
torch.save(model, 'model.pt')

# load model
model = torch.load('model.pt', weights_only=False)

# save weights
torch.save(model.state_dict(), 'model.pth') 
# model.state_dict(), using dictionary to represent parameters of model
# load weights
model.load_state_dict(torch.load('model.pth', weights_only=True))

# show state_dict of each dimension 
print("The state_dict of each layer:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
from skimage import io
from skimage.transform import resize
import numpy as np
import os

# 取得當前腳本所在目錄
script_dir = os.path.dirname(os.path.abspath(__file__))

for i in range(10):
    uploaded_file = os.path.join(script_dir, 'myDigits', f'{i}.png')
    
    # 檢查檔案是否存在
    if not os.path.exists(uploaded_file):
        print(f'Warning: File not found - {uploaded_file}')
        continue
        
    image1 = io.imread(uploaded_file, as_gray=True)

    # resize to (28, 28) 
    image_resized = resize(image1, (28, 28), anti_aliasing=True)    
    X1 = image_resized.reshape(1,28, 28) 

    # reverse to color (255->0, 0->255)
    X1 = torch.FloatTensor(1.0-X1).to(device)

    # predict
    predictions = torch.softmax(model(X1), dim=1)
    print(f'actual/prediction: {i} {np.argmax(predictions.detach().cpu().numpy())}')

print(model)

# Show summary of model
for name, module in model.named_children():
    print(f'{name}: {module}')

from torchinfo import summary
print(summary(model, (60000, 28, 28))) # input dimension 
