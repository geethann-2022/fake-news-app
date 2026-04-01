from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Simple training data
texts = [
    "Government passes new law",
    "Stock market rises today",
    "Aliens landed in India",
    "Magic cure for all diseases"
]

labels = [1, 1, 0, 0]  # 1 = Real, 0 = Fake

vectorizer = CountVectorizer()
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
    app.run(debug=True)