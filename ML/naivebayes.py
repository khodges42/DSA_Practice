import numpy as np

# Step 1: Create a sample dataset
np.random.seed(0)

# Generate random data for two classes
num_samples = 100
X_class0 = np.random.randn(num_samples, 2) + np.array([-2, -2])
X_class1 = np.random.randn(num_samples, 2) + np.array([2, 2])

# Combine the data to create the dataset
X = np.vstack([X_class0, X_class1])
y = np.array([0] * num_samples + [1] * num_samples)

# Step 2: Implement Naive Bayes Algorithm
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))
        self.priors = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

# Train Naive Bayes model
nb = NaiveBayes()
nb.fit(X, y)

# Step 3: Make Predictions
y_pred = nb.predict(X)

# Step 4: Evaluate the Model
accuracy = np.mean(y_pred == y)
print("Naive Bayes Accuracy:", accuracy)

# Visualize the results
import matplotlib.pyplot as plt

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

# Plot points with their predicted classes
plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], color='cyan', marker='x', label='Predicted Class 0')
plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], color='green', marker='x', label='Predicted Class 1')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Naive Bayes with NumPy')
plt.legend()
plt.show()
