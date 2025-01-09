import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("data/parkinsons.csv")

# Quick view
print(df.head())

# Correlation heatmap
plt.figure(figsize=(8,6))
corr = df.drop(["name"], axis=1, errors="ignore").corr()
sns.heatmap(corr, cmap="viridis")
plt.title("Correlation Heatmap")
plt.show()

# Features and target
X = df.drop(["name", "status"], axis=1, errors="ignore")
y = df["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy(0)", "Parkinson(1)"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=["Healthy(0)", "Parkinson(1)"],
            yticklabels=["Healthy(0)", "Parkinson(1)"])
plt.title("Confusion Matrix")
plt.show()