import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Set the hyperparameters
latent_dim = 1
data_dim = 4
batch_size = 48
epochs = 10000
lr = 0.0002

# Generate real-world data: (x, y) pairs
# def real_data_sampler(num_samples):
#     x = np.linspace(-1, 1, num_samples)
#     y = x**2
#     data = np.vstack((x, y)).T
#     return data

def sample_real(batch_size):

    a = torch.randint(-5, 6, (batch_size, 1)).float()
    a[a == 0] = 1       
    b = torch.randint(-5, 6, (batch_size, 1)).float()
    c = torch.randint(-5, 6, (batch_size, 1)).float()
    d = (b*c+1)/a

    # Flatten into (batch, 4): [a, b, c, d]
    return torch.cat([a, b, c, d], dim=1)

import numpy as np
# a = [[1], [2]]
# b = [[3], [4]]
# c = [[5], [6]]
# d = [[7], [8]]

# Stacked vertically
# stacked = np.vstack(((a, b), (c, d)))
# print(stacked)
# [[1 2 3]
#  [4 5 6]]

# transposed = stacked.T
# print(transposed)
# [[1 4]
#  [2 5]
#  [3 6]]

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, data_dim)
        )
    def forward(self, z):
        return self.model(z)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, data):
        return self.model(data)

generator = Generator()
discriminator = Discriminator()

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizer
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Lists to store loss values
d_losses = []
g_losses = []

# Train the GAN model
for epoch in range(epochs):
    # Training discriminator    
    optimizer_D.zero_grad()  
    # Real data
    real_data = torch.Tensor(sample_real(batch_size))
    valid = torch.ones(batch_size, 1)
    fake = torch.zeros(batch_size, 1)
    real_loss = adversarial_loss(discriminator(real_data), valid)
    # Fake data    
    z = torch.randn(batch_size, latent_dim)
    generated_data = generator(z)
    fake_loss = adversarial_loss(discriminator(generated_data), fake)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step() # Update the parameters of discriminator
    
    # Training generator    
    optimizer_G.zero_grad()
    z = torch.randn(batch_size, latent_dim)
    generated_data = generator(z)
    g_loss = adversarial_loss(discriminator(generated_data), valid)   
    g_loss.backward()
    optimizer_G.step() # Update the parameters of generator

    # Record losses
    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

    # Outputs training progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
plt.plot(g_losses, label='Generator Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Loss Curves')
plt.legend()
plt.grid(True)
plt.show()

# Plot the resulting data points
z = torch.randn(200, latent_dim)
generated_data = generator(z).detach().numpy()
for i in range(196,200):
    print(generated_data[i], "det:", (generated_data[i][0]*generated_data[i][3])-(generated_data[i][1]*generated_data[i][2]))
real_data = sample_real(200).numpy()

plt.scatter(real_data[:, 0], real_data[:, 1], color='red', label='Real data')
plt.scatter(generated_data[:, 0], generated_data[:, 1], color='blue', label='Generated data')
plt.legend()
plt.show()
