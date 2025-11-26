import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
lr = 0.0002
num_epochs = 30
noise_dim = 100
num_classes = 10  # MNIST has 10 digit classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


# Data Preparation: MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)

# Create Labeled and Unlabeled Data Subsets
labeled_indices = np.random.choice(len(full_dataset), 1000, replace=False)  # Only 1000 labeled samples
unlabeled_indices = list(set(range(len(full_dataset))) - set(labeled_indices))

labeled_loader = DataLoader(Subset(full_dataset, labeled_indices), batch_size=batch_size, shuffle=True)
unlabeled_loader = DataLoader(Subset(full_dataset, unlabeled_indices), batch_size=batch_size, shuffle=True)

# Test Loader
test_dataset = datasets.MNIST(root="../data", train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===== Define Models =====
# DCGAN Discriminator with dual outputs (like SGAN structure)
class DCGAN_Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64, num_classes=10):
        super(DCGAN_Discriminator, self).__init__()
        # Shared convolutional feature extractor
        self.shared = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 1 x 1
        )
        
        # Real/Fake classification head
        self.real_fake_output = nn.Sequential(
            nn.Linear(ndf * 8, 1),
            nn.Sigmoid()
        )
        
        # Class prediction head (num_classes + 1 for fake class)
        self.class_output = nn.Linear(ndf * 8, num_classes + 1)

    def forward(self, x):
        # Extract shared features
        shared_features = self.shared(x)
        shared_features = shared_features.view(shared_features.size(0), -1)
        
        # Real/Fake prediction
        real_fake = self.real_fake_output(shared_features).squeeze(1)
        
        # Class prediction
        class_logits = self.class_output(shared_features)
        
        return real_fake, class_logits

netD = DCGAN_Discriminator().to(device)


# DCGAN Generator
class DCGAN_Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super(DCGAN_Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 7 x 7
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 14 x 14
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )

    def forward(self, z):
        return self.main(z)

netG = DCGAN_Generator().to(device)




# Initialize Models
D = DCGAN_Discriminator().to(device)
G = DCGAN_Generator().to(device)

# Optimizers
optimizer_D = optim.Adam(D.parameters(), lr=lr)
optimizer_G = optim.Adam(G.parameters(), lr=lr)

# Loss Functions
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

d_loss_history = []
g_loss_history = []

# ===== Training Loop =====
for epoch in range(num_epochs):
    D.train()
    G.train()
    total_d_loss = 0
    total_g_loss = 0
    for (labeled_data, labels), (unlabeled_data, _) in zip(labeled_loader, unlabeled_loader):
        labeled_data, labels = labeled_data.to(device), labels.to(device)
        unlabeled_data = unlabeled_data.to(device)

        # ----- Train Discriminator -----
        optimizer_D.zero_grad()
        
        # Real labeled data - 真實且有標籤的資料
        real_pred, class_logits = D(labeled_data)
        real_labels_d = torch.ones_like(real_pred).to(device)
        d_real_loss = bce_loss(real_pred, real_labels_d) + ce_loss(class_logits, labels)

        # Unlabeled real data - 真實但無標籤的資料
        real_pred_u, _ = D(unlabeled_data)
        real_labels_u = torch.ones_like(real_pred_u).to(device)
        d_unlabeled_loss = bce_loss(real_pred_u, real_labels_u)

        # Fake data - 生成的假資料
        z = torch.randn(unlabeled_data.size(0), noise_dim, 1, 1).to(device)
        fake_data = G(z)
        fake_pred, fake_class_logits = D(fake_data.detach())
        fake_labels_d = torch.zeros_like(fake_pred).to(device)
        # 假圖像的類別標籤為 num_classes (第11類)
        fake_class_labels = torch.full((unlabeled_data.size(0),), num_classes, dtype=torch.long).to(device)
        d_fake_loss = bce_loss(fake_pred, fake_labels_d) + ce_loss(fake_class_logits, fake_class_labels)

        # Total Discriminator Loss
        d_loss = d_real_loss + d_unlabeled_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # ----- Train Generator -----
        optimizer_G.zero_grad()
        
        z = torch.randn(unlabeled_data.size(0), noise_dim, 1, 1).to(device)
        fake_data = G(z)
        fake_pred, _ = D(fake_data)
        real_labels_g = torch.ones_like(fake_pred).to(device)
        g_loss = bce_loss(fake_pred, real_labels_g)

        g_loss.backward()
        optimizer_G.step()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()

    # Record average losses for this epoch
    d_loss_history.append(total_d_loss / len(labeled_loader))
    g_loss_history.append(total_g_loss / len(labeled_loader))

    # Print Epoch Progress
    print(f"Epoch [{epoch + 1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(d_loss_history, label="Discriminator Loss")
plt.plot(g_loss_history, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.show()

# ===== Evaluate Model and Visualize Predictions =====
D.eval()
correct = 0
total = 0
examples = []  # To store examples for visualization

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        _, class_logits = D(data)
        pred = class_logits[:, :-1].argmax(dim=1)  # Ignore the "fake" class
        correct += (pred == target).sum().item()
        total += target.size(0)

        # Store some examples for visualization
        if len(examples) < 10:  # Show up to 10 examples
            examples.append((data.cpu(), pred.cpu(), target.cpu()))


accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Visualize prediction results
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()
for i, (img_batch, pred_batch, target_batch) in enumerate(examples):
    for img, pred, target, ax in zip(img_batch, pred_batch, target_batch, axes):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Pred: {pred.item()}\nTrue: {target.item()}")
        ax.axis("off")
plt.tight_layout()
plt.show()
