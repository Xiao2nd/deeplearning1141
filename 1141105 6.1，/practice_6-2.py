
'''
Given an input sequence:
x=[2, 4, 7, 11, 16]  The goal is for the RNN model to learn how to output the difference between consecutive elements, i.e., the increment at each step: y=[4−2, 7−4, 11−7, 16−11]=[2, 3, 4, 5]. In other words, the model should learn to capture how much the value changes at each step in the sequence.
Use your model to test the result of [3, 7, 8, 10, 17, 19, 25]. (The answer should be close to [4, 1, 2, 7, 2, 6]
'''
import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(0)

# Create the data ===

x_data = [
    [2, 4, 7, 11, 16, 0, 0],          
    [1, 3, 6, 10, 15, 0, 0],          
    [5, 8, 12, 17, 23, 0, 0],         
    [10, 12, 15, 19, 24, 0, 0],      
    [0, 5, 7, 10, 18, 0, 0],         
    [3, 8, 9, 12, 19, 0, 0],         
    [1, 6, 8, 11, 18, 0, 0],          
    [2, 7, 9, 12, 20, 0, 0],          
    [5, 10, 11, 14, 21, 0, 0],      
    [4, 8, 9, 11, 18, 0, 0],         
    [3, 7, 8, 10, 17, 0, 0],        
    [1, 5, 6, 8, 15, 0, 0],           
    [2, 6, 7, 9, 16, 18, 24],         
    [1, 5, 6, 8, 15, 17, 23],       
    [0, 4, 5, 7, 14, 16, 22],         
    [3, 7, 8, 10, 17, 19, 25],       
]

y_data = [
    [2, 3, 4, 5, 0, 0],
    [2, 3, 4, 5, 0, 0],
    [3, 4, 5, 6, 0, 0],
    [2, 3, 4, 5, 0, 0],
    [5, 2, 3, 8, 0, 0],
    [5, 1, 3, 7, 0, 0],
    [5, 2, 3, 7, 0, 0],
    [5, 2, 3, 8, 0, 0],
    [5, 1, 3, 7, 0, 0],
    [4, 1, 2, 7, 0, 0],
    [4, 1, 2, 7, 0, 0],
    [4, 1, 2, 7, 0, 0],
    [4, 1, 2, 7, 2, 6],
    [4, 1, 2, 7, 2, 6],
    [4, 1, 2, 7, 2, 6],
    [4, 1, 2, 7, 2, 6],
]

x_seq = torch.tensor(x_data, dtype=torch.float32)  # shape: (batch=16, seq_len=7)
y_seq = torch.tensor(y_data, dtype=torch.float32)  # shape: (batch=16, seq_len=6)
test_seq = torch.tensor([[3, 7, 8, 10, 17, 19, 25]], dtype=torch.float32)

# PyTorch RNN expects input shape: (seq_len, batch, input_size)
x_seq = x_seq.unsqueeze(-1).permute(1, 0, 2)  # (7, 16, 1)
y_seq = y_seq.unsqueeze(-1).permute(1, 0, 2)  # (6, 16, 1)
test_seq = test_seq.unsqueeze(-1).permute(1, 0, 2)  # (7, 1, 1)

# Define the model ===
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, h = self.rnn(x)     # out: (seq_len, batch, hidden_size)
        out = self.fc(out)       # Map hidden states to output: (seq_len, batch, output_size)
        return out[1:]           

model = SimpleRNN()

from torchinfo import summary
print(summary(model))
print()

# Define loss and optimizer ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ===  Training loop ===
for epoch in range(10000):
    optimizer.zero_grad()
    output = model(x_seq)
    loss = criterion(output, y_seq)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch+1}, Loss = {loss.item():.6f}")

# Test ===
with torch.no_grad():
    pred = model(test_seq).squeeze()
    print("\n Prediction:", pred.tolist())
