import numpy as np

# Step 1: Create a sample dataset
# Let's assume a linear relationship: y = 2x + 1 with some noise
np.random.seed(0)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with noise

# Step 2: Implement Linear Regression using the normal equation
# Add a bias term (column of ones) to the feature matrix
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Calculate weights using the normal equation: theta = (X_b.T * X_b)^-1 * X_b.T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Step 3: Make predictions using the learned model
def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(theta)

# Predict on the training data
y_pred = predict(X, theta_best)

# Step 4: Evaluate the model using Mean Squared Error (MSE)
mse = np.mean((y_pred - y) ** 2)
print("Weights (theta):", theta_best.flatten())
print("Mean Squared Error:", mse)

# Visualize the results
import matplotlib.pyplot as plt

plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_pred, color="red", label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with NumPy")
plt.legend()
plt.show()
