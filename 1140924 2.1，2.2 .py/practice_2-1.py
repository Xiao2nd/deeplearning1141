import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(16) 
# Generate random data 
def generate_cubic_data(num_samples):
    x = torch.linspace(-5, 5, num_samples)
    y = x**3 + 5 * x**2 - 3 * x - 7 + torch.randn(num_samples) * 2
    return x, y

# Define a cubic function model
def cubic_model(x, params):
    a, b, c, d = params
    return a * x**3 + b * x**2 + c * x + d

# Define the loss function (mean squared error)
def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

# Gradient descent optimization
def gradient_descent(x, y, learning_rate, num_epochs):
    # Initialize model parameters randomly
    params = torch.randn(4, requires_grad=True)
    # params = torch.tensor([0.83, 4.56, 0.07, -0.25], requires_grad=True)

    for epoch in range(num_epochs):
        # Calculate the model predictions
        y_pred = cubic_model(x, params)
        # Calculate the loss function
        loss = mean_squared_error(y, y_pred)
        # Calculate gradients and update parameters
        loss.backward()
        with torch.no_grad():
            params -= learning_rate * params.grad
        # Clear the gradient
        params.grad.zero_()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return params

# Generate data
num_samples = 100
x, y = generate_cubic_data(num_samples)

learning_rate = 0.0001
num_epochs = 1000
optimal_params = gradient_descent(x, y, learning_rate, num_epochs)


# Plot the fitted curve
x_test = torch.linspace(-5, 5, 100)
with torch.no_grad():
    y_pred = cubic_model(x_test, optimal_params)

plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), label='Real data')
plt.plot(x_test.numpy(), y_pred.numpy(), 'r-', label='Fit curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Output fitted parameters
print(f"Fitted parameters：a={optimal_params[0]:.2f}, b={optimal_params[1]:.2f}, c={optimal_params[2]:.2f}, d={optimal_params[3]:.2f}")

plt.show()

