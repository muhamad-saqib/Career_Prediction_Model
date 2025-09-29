from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained objects
model = joblib.load("career_model.pkl")
career_encoder = joblib.load("career_encoder.pkl")
mlb_skills = joblib.load("skills_encoder.pkl")
mlb_interests = joblib.load("interests_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Home route → HTML frontend
@app.route("/")
def home():
    return render_template("index.html")

# Predict route → JSON response
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        matric = float(data["Matric Percentage"])
        inter = float(data["Intermediate Percentage"])
        skills = [s.strip().lower() for s in data["Skills"]]
        interests = [i.strip().lower() for i in data["Interests"]]

        scaled = scaler.transform([[matric, inter]])[0]
        skills_vec = mlb_skills.transform([skills])[0]
        interests_vec = mlb_interests.transform([interests])[0]

        features = np.concatenate([scaled, skills_vec, interests_vec]).reshape(1, -1)
        pred = model.predict(features)[0]
        career = career_encoder.inverse_transform([pred])[0]

        return jsonify({"predicted_career": career})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
