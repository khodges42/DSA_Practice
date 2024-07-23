import numpy as np
from collections import Counter

# Step 1: Create a sample dataset
np.random.seed(0)

# Generate random data
num_samples = 100
X = np.random.rand(num_samples, 2)  # 100 samples, 2 features

# Generate labels for two classes
y = np.array([0 if x[0] + x[1] < 1 else 1 for x in X])

# Step 2: Implement k-NN Algorithm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        y_pred.append(most_common[0][0])
    return np.array(y_pred)

# Step 3: Make Predictions
# Split the dataset into train and test sets
train_ratio = 0.8
train_size = int(num_samples * train_ratio)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Predict on the test data
k = 3
y_pred = knn_predict(X_train, y_train, X_test, k)

# Step 4: Evaluate the Model
accuracy = np.mean(y_pred == y_test)
print("k-NN Accuracy:", accuracy)

# Visualize the results
import matplotlib.pyplot as plt

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

# Plot test points with their predicted classes
plt.scatter(X_test[y_pred == 0][:, 0], X_test[y_pred == 0][:, 1], color='cyan', marker='x', label='Predicted Class 0')
plt.scatter(X_test[y_pred == 1][:, 0], X_test[y_pred == 1][:, 1], color='green', marker='x', label='Predicted Class 1')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('k-Nearest Neighbors with NumPy')
plt.legend()
plt.show()