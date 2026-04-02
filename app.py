from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

app = Flask(__name__)

# Simple training data
texts = [
    # ✅ REAL NEWS (1)
    "Government announces new education policy",
    "Stock market shows positive growth today",
    "Scientists discovered a new species",
    "India successfully launches satellite",
    "Technology is improving rapidly",
    "Health experts recommend regular exercise",
    "New law passed for road safety",
    "University releases exam schedule",
    "Company introduces new smartphone",
    "Rainfall expected in southern regions",

    # ❌ FAKE NEWS (0)
    "Aliens landed in India yesterday",
    "Magic pill cures all diseases instantly",
    "Man becomes invisible after experiment",
    "Time travel machine discovered in lab",
    "Drinking water turns people into superheroes",
    "Zombie outbreak reported in city",
    "Humans can live without oxygen",
    "Flying cars available for free",
    "Miracle cure found for all problems overnight",
    "Earth will disappear tomorrow"
]

labels = [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    X_input = vectorizer.transform([news])
    result = model.predict(X_input)

    output = "Real News ✅" if result[0] == 1 else "Fake News ❌"
    return render_template('index.html', prediction=output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)