import numpy as np

# Step 1: Create a sample dataset
np.random.seed(0)

# Generate random data
num_samples = 100
X = 2 * np.random.rand(num_samples, 2) - 1  # 100 samples, 2 features
true_weights = np.array([2, -3])  # True weights
bias = 0.5  # Bias term

# Generate labels
linear_combination = X.dot(true_weights) + bias
y = (linear_combination > 0).astype(int)  # Binary labels

# Step 2: Implement Logistic Regression using Gradient Descent
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
    weights = np.zeros(n + 1)  # Initialize weights

    for _ in range(iterations):
        linear_model = X_b.dot(weights)
        y_pred = sigmoid(linear_model)
        error = y_pred - y
        gradients = X_b.T.dot(error) / m
        weights -= learning_rate * gradients

    return weights

# Train logistic regression model
weights = logistic_regression(X, y)

# Step 3: Make Predictions
def predict(X, weights):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    linear_model = X_b.dot(weights)
    y_pred = sigmoid(linear_model)
    return (y_pred >= 0.5).astype(int)

# Predict on the training data
y_pred = predict(X, weights)

# Step 4: Evaluate the Model
accuracy = np.mean(y_pred == y)
print("Weights:", weights)
print("Accuracy:", accuracy)

# Visualize the results
import matplotlib.pyplot as plt

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

# Plot decision boundary
x_values = np.linspace(-1, 1, 100)
decision_boundary = -(weights[0] + weights[1] * x_values) / weights[2]
plt.plot(x_values, decision_boundary, label='Decision Boundary', color='green')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression with NumPy')
plt.legend()
plt.show()