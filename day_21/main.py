from fastapi import FastAPI
from pydantic import BaseModel
import pickle


# Initialize the FastAPI app.
app = FastAPI(title="Fake News Detection API", version="1.0")

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as tfidf_vectorizer_file:
    tfidf_vectorizer = pickle.load(tfidf_vectorizer_file)

# Define the basic structure of ur body using Pydantic
class NewsArticle(BaseModel):
    text: str

# Define the prediction endpoint.
@app.post("/predict")
def make_prediction(article: NewsArticle):
    # Vectorize the incoming text.
    vectorized_text = tfidf_vectorizer.transform([article.text])

    # Make prediction
    prediction = model.predict(vectorized_text)

    # Return the result.
    result = "Fake" if prediction[0] == 0 else "True"

    return {"prediction": result}


# Build the Home page of our application.
@app.get("/")
def index():
    return {"message": "Welcome to the Fake News Detection API! Visit /docs for Swagger documentation."}
