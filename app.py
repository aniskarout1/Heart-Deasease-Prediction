from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

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
        if prediction[0] == 0:
            result = "You are healthy"
        else:
            result = "You should consult your doctor"
        return render_template("index.html", prediction_text=result)
    except:
        return render_template("index.html", prediction_text="Invalid input. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
