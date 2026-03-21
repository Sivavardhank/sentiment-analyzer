# sentiment_model.py

import pandas as pd
import re
import pickle

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ------------------------
# 1️⃣ LOAD DATA
# ------------------------
df = pd.read_csv("IMDB Dataset.csv")

# Convert labels
df["sentiment"] = df["sentiment"].map({"positive":1, "negative":0})

# ------------------------
# 2️⃣ ADD EXTRA DATA (IMPORTANT 🔥)
# ------------------------
extra_data = pd.DataFrame({
    "review": [
        "not bad movie",
        "not bad at all",
        "not bad I liked it",
        "not good movie",
        "not good at all",
        "not good I hate it"
    ],
    "sentiment": [1,1,1,0,0,0]
})

df = pd.concat([df, extra_data], ignore_index=True)

# ------------------------
# 3️⃣ CLEAN TEXT
# ------------------------
def clean_text(text):
    text = text.lower()
    
    # 🔥 IMPORTANT RULES
    text = text.replace("not bad", "good")
    text = text.replace("not good", "bad")
    
    text = re.sub(r'[^a-z ]', ' ', text)
    return text

df["review"] = df["review"].apply(clean_text)

# ------------------------
# 4️⃣ FEATURE EXTRACTION
# ------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,3)   # 🔥 key improvement
)

X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

print("Shape:", X.shape)

# ------------------------
# 5️⃣ TRAIN TEST SPLIT
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# 6️⃣ LOGISTIC REGRESSION
# ------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall   :", recall_score(y_test, lr_pred))

# ------------------------
# 7️⃣ NAIVE BAYES
# ------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\n--- Naive Bayes ---")
print("Accuracy :", accuracy_score(y_test, nb_pred))
print("Precision:", precision_score(y_test, nb_pred))
print("Recall   :", recall_score(y_test, nb_pred))

# ------------------------
# 8️⃣ CHOOSE BEST MODEL
# ------------------------
if accuracy_score(y_test, nb_pred) > accuracy_score(y_test, lr_pred):
    model = nb_model
    print("\nUsing Naive Bayes for deployment")
else:
    model = lr_model
    print("\nUsing Logistic Regression for deployment")

# ------------------------
# 9️⃣ TEST CUSTOM INPUT
# ------------------------
tests = [
    "not bad movie",
    "worst film ever",
    "this movie is amazing",
    "i hate this movie"
]

print("\n--- Testing ---")
for t in tests:
    t_clean = [clean_text(t)]
    t_vec = vectorizer.transform(t_clean)
    pred = model.predict(t_vec)[0]

    if pred == 1:
        print(f"{t} → Positive 😊")
    else:
        print(f"{t} → Negative 😡")

# ------------------------
# 🔟 SAVE MODEL
# ------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
