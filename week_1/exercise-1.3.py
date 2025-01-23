#
# He initialized weights to start
# Standardized inputs and targets
#
import numpy as np
import matplotlib.pyplot as plt

######
print("1. Prepare training data")
input_size = 1
x = np.array([1, 2, 3, 4, 5])
targets = np.array([2, 4, 6, 8, 10])

print("1a. Normalize x")
mean_x = np.mean(x)
std_x = np.std(x)
print(f"mean_x: {mean_x}")
print(f"std_x: {std_x}")

mean_targets = np.mean(targets)
std_targets = np.std(targets)
print(f"mean_targets: {mean_targets}")
print(f"std_targets: {std_targets}")

x = (x - mean_x) / std_x
targets = (targets - mean_targets) / std_targets

print(f"x: {x}")
print(f"targets: {targets}")

######
print("")
print("2. Initialize model parameters")
W = np.random.randn() * np.sqrt(2 / input_size) # single weight for simplicity
b = 0                                           # Bias

print(f"W: {W}")
print(f"b: {b}")

######
print("")
print("3. Define the prediction function")
y_pred = W*x + b
print(f"y_pred: {y_pred}")

######
print("")
print("4. Calculate the loss")
loss_mse = np.mean((y_pred - targets) ** 2)
print(f"loss_mse: {loss_mse}")

loss_m_1_8e = np.mean(np.abs((y_pred - targets)) ** 1.8)
print(f"loss_m_1_8e: {loss_m_1_8e}")

######
print("")
print("5. Compute gradients")
dW = np.mean(2 * (y_pred - targets) * x)  # Gradient w.r.t. W
db = np.mean(2 * (y_pred - targets))      # Gradient w.r.t. b
print(f"dW: {dW}")
print(f"db: {db}")

######
print("")
print("6. Update parameters")
lr = 0.01
W -= lr * dW
b -= lr * db
print(f"W: {W}")
print(f"b: {b}")


######
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

######
print("")
print("8. Test the model")
x_test = np.array([6, 7])
x_test = (x_test - mean_x) / std_x

print(f"Epochs: {epochs}")
print(f"Learning Rate: {lr}")
print(f"x_test: {x_test}")
y_test_pred = W * x_test + b
print("Predictions:", y_test_pred)

y_test_pred = (y_test_pred * std_targets) + mean_targets
print("Scaled Predictions:", y_test_pred)


