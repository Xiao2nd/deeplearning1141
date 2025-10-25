import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

torch.manual_seed(16)

# Generate random data 
def generate_cubic_data(num_samples):
    x = torch.linspace(-5, 5, num_samples)
    y = x**3 + 5 * x**2 - 3 * x - 7 + torch.randn(num_samples) * 2
    return x, y


#Define a simple neural network model
class cubicModel(nn.Module):
    def __init__(self):
        super(cubicModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
        self.fc2 = nn.Linear(10, 1)  # Hide layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Create a model
model = cubicModel()

# Create an optimizer
criterion = nn.MSELoss()  # Mean squared error loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.07)

num_samples = 100
x, y = generate_cubic_data(num_samples)
print (x.shape)
print (y.shape)

# Train the model
num_epochs = 100000
for epoch in range(num_epochs):
    # Forward propagation
    outputs = model(x.unsqueeze(1)) #Increase dimensions
    loss = criterion(outputs, y.unsqueeze(1)) #Increase dimensions
    # or loss = criterion(outputs.squeeze(), y)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the fitted curve
model.eval()
with torch.no_grad():
    x_test = torch.linspace(-5, 5, 100)
    y_pred = model(x_test.unsqueeze(1))

plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), label='Real data')
plt.plot(x_test.numpy(), y_pred.numpy(), 'r-', label='Fit curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()