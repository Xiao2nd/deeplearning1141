import torch
from torchviz import make_dot

x = torch.tensor((1,2,3),dtype=torch.float32,requires_grad=True)
y = x**4
z = y*2

def mean(n): # avg of a list
    return sum(n) / len(n)

if __name__ == "__main__":
    u = mean(x)
    dot = make_dot(u)
    dot.view()
    print(u)