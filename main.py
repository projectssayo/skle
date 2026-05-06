from fastapi import FastAPI
import joblib

app = FastAPI()

model = None
vectorizer = None

@app.on_event("startup")
def load_model():
    global model, vectorizer
    model, vectorizer = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(text: str):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    return {
        "text": text,
        "label": "neutral" if pred == 1 else "non-neutral"
    }
