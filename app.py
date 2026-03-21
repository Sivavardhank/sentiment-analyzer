# app.py
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon')
except:
    nltk.download('vader_lexicon')
    
from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download once
nltk.download('vader_lexicon')

app = Flask(__name__)

# Load VADER
sia = SentimentIntensityAnalyzer()

# ------------------------
# ROUTE
# ------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = ""

    if request.method == "POST":
        msg = request.form["message"]

        scores = sia.polarity_scores(msg)
        compound = scores['compound']

        # Prediction logic
        if compound >= 0.05:
            prediction = "Positive 😊"
        elif compound <= -0.05:
            prediction = "Negative 😡"
        else:
            prediction = "Neutral 😐"

        confidence = round(abs(compound) * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)

# ------------------------
# RUN APP
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
