from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

texts = [
    "I love this product",
    "This is amazing",
    "You are stupid",
    "I hate you",
    "This is okay",
    "Nothing special",
    "You idiot",
    "Great work",
    "Terrible experience",
    "Very nice"
]

labels = [
    1,  # neutral/positive
    1,
    0,  # toxic / non-neutral
    0,
    1,
    1,
    0,
    1,
    0,
    1
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

joblib.dump((model, vectorizer), "model.pkl")

print("Model ready")
