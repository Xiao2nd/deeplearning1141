import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Fix the random seed for reproducibility
torch.manual_seed(16)

# Generate synthetic data: y = x**3 + 5* x**2 -3*x -7 + noise
def generate_data(num_samples):
    x = torch.linspace(-5, 5, num_samples)
    y = x**3 + 5 * x**2 - 3 * x - 7 + torch.randn(num_samples) * 2
    return x, y

# Define a neural network that learns coefficients [a, b, c]
class CoefficientLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)   # Output: coefficients [a, b, c, d]
        )
        
    def forward(self, dummy_input):
        return self.net(dummy_input)  # Use dummy input to trigger forward pass

# Initialize data and model
num_samples = 100
x, y = generate_data(num_samples)
model = CoefficientLearner()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.07)

# Dummy input is used only to trigger forward pass for coefficient generation
dummy_input = torch.tensor([[0.0]])

# Train the model
for epoch in range(10000):
    coeffs = model(dummy_input)         # shape: [1, 4]
    a, b, c, d = coeffs[0]              # Extract coefficients
    y_pred = a * x**3 + b * x**2 + c * x + d   # Predict y using the polynomial

    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# Plot the results
model.eval()
with torch.no_grad():
    coeffs = model(dummy_input)[0]
    a, b, c, d = coeffs
    x_test = torch.linspace(-5, 5, 100)
    y_fit = a * x_test**3 + b * x_test**2 + c * x_test + d

plt.scatter(x.numpy(), y.numpy(), label="Real data")
plt.plot(x_test.numpy(), y_fit.numpy(), 'r-', label="Fit curve")
plt.legend()
print(f"Fitted: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
plt.show()
