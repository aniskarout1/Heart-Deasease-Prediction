# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("heart_disease_data.csv")
x = df.drop(columns="target", axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

model = RandomForestClassifier(n_estimators=40)
model.fit(x_train, y_train)

# Save the model
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)
