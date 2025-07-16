# Imports
import os
import numpy as np
import pandas as pd
from collections import Counter

# Verify input directory contents (Kaggle-style)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Update path to match local or Kaggle if needed
TRAIN_PATH = '../Iris project/data/iris_train.csv'
TEST_PATH  = '../Iris project/data/iris_test.csv'

# Training Algorithm: Learn class centers using manual gradient-like update
def get_minimizer(Y, X, iterations, learning_rate):
    tempY = {"a": X.min(), "b": X.max(), "c": X.mean()}
    total = len(X)
    for _ in range(iterations):
        for i in range(total):
            a = abs(X[i] - tempY["a"])
            b = abs(X[i] - tempY["b"])
            c = abs(X[i] - tempY["c"])
            if a < b and a < c:
                answer = "a"
            elif b < a and b < c:
                answer = "b"
            else:
                answer = "c"

            if answer != Y[i]:
                tempY[Y[i]] += learning_rate * (X[i] - tempY[Y[i]])
            else:
                tempY[Y[i]] += learning_rate * (X[i] - tempY[Y[i]]) / 5
    return tempY

# Evaluation Function
def test(Y, data, finalY):
    columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    total = len(Y)
    right = 0
    true_right = 0
    for i in range(total):
        points = {'a': 0, 'b': 0, 'c': 0}
        for idx, col in enumerate(columns):
            X = data[col].tolist()
            params = finalY[idx]
            dists = {
                'a': abs(X[i] - params["a"]),
                'b': abs(X[i] - params["b"]),
                'c': abs(X[i] - params["c"])
            }
            sorted_labels = sorted(dists, key=lambda x: dists[x])
            points[sorted_labels[0]] += 2
            points[sorted_labels[1]] += 1

        max_points = max(points.values())
        best_labels = [label for label, score in points.items() if score == max_points]
        predicted = " or ".join(best_labels)
        is_right = Y[i] in best_labels
        is_true_right = (is_right and len(best_labels) == 1)
        result = "right" if is_right else "wrong"
        either = f" (either {predicted})" if len(best_labels) > 1 else ""
        print(f"row {i}: '{predicted}' vs '{Y[i]}' → {result}{either} | points: {points}")
        if is_right:
            right += 1
        if is_true_right:
            true_right += 1
    print(f"\nTotal correct: {right}/{total} | Accuracy: {right / total:.2%}")
    print(f"Total true correct: {true_right}/{total} | Accuracy: {true_right / total:.2%} (the either become false)")

# Load and Prepare Training Data
data = pd.read_csv(TRAIN_PATH)
data['target'] = data['target'].map({0: 'a', 1: 'b', 2: 'c'})
Y = data['target']

# Train on each feature dimension
finalY = [
    get_minimizer(Y, data['sepal length (cm)'], 100, 0.01),
    get_minimizer(Y, data['sepal width (cm)'], 100, 0.01),
    get_minimizer(Y, data['petal length (cm)'], 100, 0.01),
    get_minimizer(Y, data['petal width (cm)'], 100, 0.01)
]
print("Final learned centers:", finalY)

# Evaluate on training data
test(Y, data, finalY)

# Predict Test Set
def predict_and_return_df(test_data, finalY):
    columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    predictions = []
    for i in range(len(test_data)):
        points = {'a': 0, 'b': 0, 'c': 0}
        for idx, col in enumerate(columns):
            X = test_data[col]
            params = finalY[idx]
            dists = {
                'a': abs(X.iloc[i] - params["a"]),
                'b': abs(X.iloc[i] - params["b"]),
                'c': abs(X.iloc[i] - params["c"])
            }
            sorted_labels = sorted(dists, key=lambda x: dists[x])
            points[sorted_labels[0]] += 2
            points[sorted_labels[1]] += 1
        best_label = max(points, key=lambda x: (points[x], -ord(x[0])))
        predictions.append(best_label)
    result_df = test_data.copy()
    result_df['predicted'] = predictions
    result_df['predicted_numeric'] = result_df['predicted'].map({'a': 0, 'b': 1, 'c': 2})
    return result_df

# Load test data
test_data = pd.read_csv(TEST_PATH)

# Predict and display
predicted_df = predict_and_return_df(test_data, finalY)
print("\nPrediction Results:")
print(predicted_df)

