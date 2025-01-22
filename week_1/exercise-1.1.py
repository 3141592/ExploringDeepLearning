import numpy as np
import matplotlib.pyplot as plt

print("1. Prepare training data")
x = np.array([1, 2, 3, 4, 5])
targets = np.array([2, 4, 6, 8, 10])

print("1a. Normalize x")
x = x / 5
targets = targets / 10
print(f"x: {x}")
print(f"targets: {targets}")

print("")
print("2. Initialize model parameters")
W = np.random.randn() # single weight for simplicity
b = np.random.randn() # Bias
W = 0.8997917141193307
b = 2.1376684286193584

print(f"W: {W}")
print(f"b: {b}")

print("")
print("3. Define the prediction function")
y_pred = W*x + b
print(f"y_pred: {y_pred}")

print("")
print("4. Calculate the loss")
loss_mse = np.mean((y_pred - targets) ** 2)
print(f"loss_mse: {loss_mse}")

loss_m_1_8e = np.mean(np.abs((y_pred - targets)) ** 1.8)
print(f"loss_m_1_8e: {loss_m_1_8e}")

print("")
print("5. Compute gradients")
dW = np.mean(2 * (y_pred - targets) * x)  # Gradient w.r.t. W
db = np.mean(2 * (y_pred - targets))      # Gradient w.r.t. b
print(f"dW: {dW}")
print(f"db: {db}")

print("")
print("6. Update parameters")
lr = 0.01
W -= lr * dW
b -= lr * db
print(f"W: {W}")
print(f"b: {b}")


print("")
print("7. Iterate")
epochs = 50
for epoch in range(epochs):
    y_pred = W * x + b
    loss = np.mean((y_pred - targets) ** 2)
    #loss_m_1_8e = np.mean(np.abs((y_pred - targets)) ** 1.8)
    dW = np.mean(2 * (y_pred - targets) * x)
    db = np.mean(2 * (y_pred - targets))
    W -= lr * dW
    b -= lr * db
    print(f"Epoch {epoch}: Loss = {loss}")
    #print(f"Epoch {epoch}: Loss_m_1_8e = {loss_m_1_8e}")

print("")
print("8. Test the model")
x_test = np.array([6, 7])
x_test = x_test / 5

print(f"Epochs: {epochs}")
print(f"Learning Rate: {lr}")
print(f"x_test: {x_test}")
y_test_pred = W * x_test + b
print("Predictions:", y_test_pred)

y_test_pred = y_test_pred * 10
print("Scaled Predictions:", y_test_pred)


