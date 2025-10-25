import torch
import torch.nn as nn

# Define a 3x3 input feature map
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 4, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# Shape after unsqueeze: [1, 2, 3], [4, 5, 6], [1, 4, 1] to fit PyTorch's expected (batch_size, channels, height, width)

# Define a transposed convolution layer with a 2x2 kernel, stride 2, and no bias
deconv = nn.ConvTranspose2d(
    in_channels=1,      # Number of input channels
    out_channels=1,     # Number of output channels
    kernel_size=2,      # Size of the convolutional kernel
    stride=2,           # Stride value to expand the output size
    bias=False          # Disable bias term
)

# Manually set the weight of the transposed convolution kernel to a simple pattern with torch.no_grad():
kernal = nn.Parameter(torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.float32))

deconv.weight = nn.Parameter(kernal)

# Perform forward propagation through the transposed convolution layer
output_tensor = deconv(input_tensor)

# Print input and output details
print("Original input feature map:")
print(input_tensor[0][0])

print("Kernel:")
print(kernal[0][0])

print("\nOutput feature map after transposed convolution:")
print(output_tensor[0][0])
