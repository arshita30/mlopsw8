import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

def poison_data(X, poison_percent):
    X_poisoned = X.copy()
    n_poison = int(len(X_poisoned) * poison_percent)
    poison_indices = np.random.choice(X_poisoned.index, size=n_poison, replace=False)

    for col in X_poisoned.columns:
        col_mean = X_poisoned[col].mean()
        col_std = X_poisoned[col].std()
        # Use loc to avoid chained assignment warning
        X_poisoned.loc[poison_indices, col] = np.random.normal(col_mean, col_std, size=n_poison)

    return X_poisoned

poison_levels = [0.05, 0.10, 0.25, 0.40, 0.50]
train_accuracies = []
val_accuracies = []

print("| Poison Level | Train Accuracy | Val Accuracy |")
print("|--------------|----------------|---------------|")

for level in poison_levels:
    X_poisoned = poison_data(X, level)
    X_train, X_val, y_train, y_val = train_test_split(X_poisoned, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    print(f"| {int(level*100)}% | {train_acc:.3f} | {val_acc:.3f} |")

# Accuracy drop summary
acc_drop = val_accuracies[0] - val_accuracies[-1]
print(f"Accuracy dropped by: {acc_drop:.3f} from 5% to 50% poisoning.")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot([int(p*100) for p in poison_levels], train_accuracies, marker='o', label="Train Accuracy")
plt.plot([int(p*100) for p in poison_levels], val_accuracies, marker='s', label="Validation Accuracy")
plt.title("Model Accuracy under Different Poisoning Levels")
plt.xlabel("Poison Level (%)")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("poisoning_accuracy_plot.png")
print("Saved plot to poisoning_accuracy_plot.png")
