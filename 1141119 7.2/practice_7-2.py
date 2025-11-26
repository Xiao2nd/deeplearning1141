import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 後端,更穩定
import matplotlib.pyplot as plt
import os
from torchvision.datasets import MNIST
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])
dataset = MNIST('../data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(dataset[0][0])

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Sets the embedding layer as input to Label
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100) 
        c = self.label_emb(labels) 
        x = torch.cat([z, c], 1) # Merge inputs
        out = self.model(x)
        return out.view(x.size(0), 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Sets the embedding layer as input to Label
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1) # Merge inputs
        out = self.model(x)
        return out.squeeze()

generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)# Random random numbers [1, 10]
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()
    # Train for real image
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    
    # Train for fake image
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device) # Random random numbers [1, 10]
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item()

num_epochs = 30
n_critic = 5
display_step = 300

# 創建保存圖片的資料夾
save_dir = os.path.join(os.path.dirname(__file__), 'generated_images')
os.makedirs(save_dir, exist_ok=True)

# 啟用互動模式並創建圖像窗口
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch))
    
    for i, (images, labels) in enumerate(data_loader):
        real_images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        
        generator.train()
        batch_size = real_images.size(0)
        
        # 訓練判別器
        d_loss = discriminator_train_step(len(real_images), discriminator,
                                          generator, d_optimizer, criterion,
                                          real_images, labels)
        
        # 訓練生成器
        g_loss = generator_train_step(batch_size, discriminator, 
                                     generator, g_optimizer, criterion)
    
    # 每個 epoch 結束後顯示和保存圖片
    generator.eval()
    print('Epoch {}: g_loss: {}, d_loss: {}'.format(epoch, g_loss, d_loss))
    
    z = Variable(torch.randn(9, 100)).to(device)
    sample_labels = Variable(torch.LongTensor(np.arange(9))).to(device)
    sample_images = generator(z, sample_labels).unsqueeze(1).data.cpu()
    grid = make_grid(sample_images, nrow=3, normalize=True)\
            .permute(1,2,0).numpy()
    
    # 清除並更新圖像
    ax.clear()
    ax.imshow(grid)
    ax.axis('off')
    ax.set_title(f'Epoch {epoch} - g_loss: {g_loss:.4f}, d_loss: {d_loss:.4f}')
    
    # 保存當前 epoch 的圖片
    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    print(f'Saved image to {save_path}')
    
    # 必須有 pause 才能更新顯示
    plt.pause(1)

# 訓練完成後保持圖像顯示
plt.ioff()
plt.show()

# Label the name
label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
z = Variable(torch.randn(100, 100)).to(device)
labels = Variable(torch.LongTensor([i for _ in range(10) for i in range(10)])).to(device)
# Generate images
sample_images = generator(z, labels).unsqueeze(1).data.cpu()
grid = make_grid(sample_images, nrow=10, normalize=True)\
                .permute(1,2,0).numpy()
# Display the image
fig, ax = plt.subplots(figsize=(15,15))
ax.imshow(grid)
plt.yticks([])
plt.xticks(np.arange(15, 300, 30), label_names, rotation=45, fontsize=20)

