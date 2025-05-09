from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load best model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_name.txt", "r") as f:
    model_name = f.read()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        features = [
            float(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            float(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]
        arr = np.array(features).reshape(1, -1)
        prediction = model.predict(arr)
        result = "🟢 You are healthy!" if prediction[0] == 0 else "🔴 You should consult your doctor."
        return render_template("index.html", prediction_text=result, model_used=model_name)
    except Exception as e:
        return render_template("index.html", prediction_text="⚠️ Invalid input. Please try again.", model_used="")

if __name__ == "__main__":
    app.run(debug=True)
