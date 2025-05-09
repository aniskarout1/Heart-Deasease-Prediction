import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("heart_disease_data.csv")
x = df.drop(columns="target", axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2
)

# Create models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=40),
    "SVC": SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression()
}

accuracies = {}
best_model = None
best_model_name = ""
best_score = 0

# Train and evaluate
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name

# Save best model
with open("heart_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model_name.txt", "w") as f:
    f.write(best_model_name)

# Save accuracies
with open("model_accuracies.json", "w") as f:
    json.dump(accuracies, f)

# ✅ Create static folder if not exist
os.makedirs("static", exist_ok=True)

# ✅ Bar Chart: Accuracy of each model
model_names = list(accuracies.keys())
scores = list(accuracies.values())
colors = ['#66b3ff' if name != best_model_name else '#ff4d4d' for name in model_names]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, scores, color=colors)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

# Annotate accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2%}",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.savefig('static/model_accuracy_bar.png')
plt.close()
