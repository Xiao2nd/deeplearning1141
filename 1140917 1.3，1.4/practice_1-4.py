import torch

def dfunc(x):
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y = x ** 3 + 1
    z = y ** 2
    z.backward()
    return x.grad

if __name__ == "__main__":
    print(dfunc(2.0))