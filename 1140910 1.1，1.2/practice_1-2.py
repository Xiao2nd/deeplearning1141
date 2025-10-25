
import matplotlib.pyplot as plt

def loss(w, b, data):
    total = 0
    for x, y in data:
        total += (w*x + b-y)**2
    return total / (2*len(data))

def updateW(w, b, data, learning_rate):
    total = 0
    for x, y in data:
        total += (w*x + b - y) * x
    return w - (learning_rate * total / len(data))

def updateB(w, b, data, learning_rate):
    total = 0
    for x, y in data:
        total += (w*x + b - y)
    return b - (learning_rate * total / len(data))

def run_loss(w, b, learning_rate, data, epochs):
    loss_record = []
    
    for epoch in range(epochs):
        current_loss = loss(w, b, data)
        loss_record.append(current_loss)

        new_w = updateW(w, b, data, learning_rate)
        new_b = updateB(w, b, data, learning_rate)
        w, b = new_w, new_b
        print(f"Epoch {epoch+1}, w: {w}, b: {b}, loss: {current_loss}")
    
    final_loss = loss(w, b, data)
    print(f"Final parameters - w: {w}, b: {b}, loss: {final_loss}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_record, 'b-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return

if __name__ == "__main__":
    
    data = [(0, 2), (2, -1), (1, 0)]
    w = -1
    b = 3
    learning_rate = 0.1
    epochs = 20

    run_loss(w, b, learning_rate, data, epochs)



