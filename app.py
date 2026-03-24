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
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    score = sia.polarity_scores(text)
    compound = score['compound']

    # 🔥 Custom thresholds (better)
    if compound >= 0.3:
        return "Positive 😊", compound
    elif compound <= -0.3:
        return "Negative 😡", compound
    else:
        return "Neutral 😐", compound

        confidence = round(abs(compound) * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)

# ------------------------
# RUN APP
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
