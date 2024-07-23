import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset
np.random.seed(0)

# Generate random data for two classes
num_samples = 200
X_class0 = np.random.randn(num_samples // 2, 2) + np.array([-2, -2])
X_class1 = np.random.randn(num_samples // 2, 2) + np.array([2, 2])

# Combine the data to create the dataset
X = np.vstack([X_class0, X_class1])
y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))

# Shuffle the data
indices = np.random.permutation(num_samples)
X = X[indices]
y = y[indices]

# Step 2: Implement a Simple Decision Tree
class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)

        # Stop if max depth is reached or all samples belong to one class
        if depth >= self.max_depth or len(set(y)) == 1:
            return {"class": predicted_class}

        # Find the best split
        best_idx, best_threshold = self._best_split(X, y)
        if best_idx is None:
            return {"class": predicted_class}

        # Recursively grow the tree
        left_indices = X[:, best_idx] < best_threshold
        right_indices = X[:, best_idx] >= best_threshold

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {"feature_index": best_idx, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        # Initialize variables to track the best split
        best_gini = float("inf")
        best_idx, best_threshold = None, None

        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]

        # Iterate over all features and thresholds to find the best split
        for feature_index in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
            num_left = [0] * len(np.unique(y))
            num_right = num_samples_per_class.copy()

            for i in range(1, m):  # i starts from 1 to ensure we have at least one sample in each side
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in np.unique(y))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in np.unique(y))

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = feature_index
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_threshold

    def predict(self, X):
        return [self._predict_one(sample, self.tree) for sample in X]

    def _predict_one(self, sample, tree):
        if "class" in tree:
            return tree["class"]

        feature_value = sample[tree["feature_index"]]
        if feature_value < tree["threshold"]:
            return self._predict_one(sample, tree["left"])
        else:
            return self._predict_one(sample, tree["right"])


# Step 3: Implement the Random Forest Algorithm
class RandomForest:
    def __init__(self, num_trees=10, max_depth=3, sample_size=0.8):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.num_trees):
            # Bootstrap sampling
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, int(n_samples * self.sample_size), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Aggregate predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)


# Train random forest model
forest = RandomForest(num_trees=10, max_depth=3, sample_size=0.8)
forest.fit(X, y)

# Step 4: Make Predictions
y_pred = forest.predict(X)

# Step 5: Evaluate the Model
accuracy = np.mean(y_pred == y)
print("Random Forest Accuracy:", accuracy)

# Visualize the results
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

# Plot points with their predicted classes
plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], color='cyan', marker='x', label='Predicted Class 0')
plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], color='green', marker='x', label='Predicted Class 1')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Random Forest with NumPy')
plt.legend()
plt.show()
