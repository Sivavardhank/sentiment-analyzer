from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Download VADER safely
try:
    nltk.data.find('sentiment/vader_lexicon')
except:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    score = sia.polarity_scores(text)
    compound = score['compound']

    # Custom thresholds
    if compound >= 0.3:
        return "Positive 😊", round(compound*100,2)
    elif compound <= -0.3:
        return "Negative 😡", round(compound*100,2)
    else:
        return "Neutral 😐", round(compound*100,2)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        prediction, confidence = predict_sentiment(text)

        return render_template("index.html",
                               prediction=prediction,
                               confidence=confidence)

    # ✅ VERY IMPORTANT (for GET request)
    return render_template("index.html",
                           prediction=None,
                           confidence=None)

if __name__ == "__main__":
    app.run(debug=True)
