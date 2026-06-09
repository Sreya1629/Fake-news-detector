from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    data = vectorizer.transform([news])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    confidence = round(max(probability) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)