import pandas as pd
import matplotlib.pyplot as plt
from src.knn import KNNClassifier
from src.distances import euclidean
from src.benchmark import benchmark_custom, benchmark_sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/iris.csv")

print("Columns in dataset:", df.columns.tolist())

# -----------------------------
# Automatically detect features and label
# -----------------------------
# Rule:
# - Features = numeric columns
# - Label = last non-numeric column

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

if len(non_numeric_cols) == 0:
    raise ValueError("No label column found (non-numeric).")

label_col = non_numeric_cols[-1]   # safest assumption
feature_cols = numeric_cols

print("Feature columns:", feature_cols)
print("Label column:", label_col)

X = df[feature_cols].values.tolist()
y = df[label_col].values.tolist()

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Custom KNN
# -----------------------------
custom_knn = KNNClassifier(
    k=5,
    distance_fn=euclidean,
    weighted=True
)

preds_custom, time_custom = benchmark_custom(
    custom_knn, X_train, y_train, X_test
)

# -----------------------------
# Sklearn benchmark (allowed only for comparison)
# -----------------------------
preds_sklearn, time_sklearn = benchmark_sklearn(
    X_train, y_train, X_test
)

# -----------------------------
# Evaluation
# -----------------------------
acc_custom = accuracy_score(y_test, preds_custom)
acc_sklearn = accuracy_score(y_test, preds_sklearn)

print(f"\nCustom KNN Accuracy  : {acc_custom:.4f}")
print(f"Sklearn KNN Accuracy : {acc_sklearn:.4f}")
print(f"Custom Runtime (s)   : {time_custom:.4f}")
print(f"Sklearn Runtime (s)  : {time_sklearn:.4f}")

# -----------------------------
# Plots
# -----------------------------
plt.figure()
plt.bar(["Custom", "Sklearn"], [acc_custom, acc_sklearn])
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.savefig("plots/accuracy.png")
plt.show()

plt.figure()
plt.bar(["Custom", "Sklearn"], [time_custom, time_sklearn])
plt.ylabel("Time (seconds)")
plt.title("Runtime Comparison")
plt.savefig("plots/runtime.png")
plt.show()
