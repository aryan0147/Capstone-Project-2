import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model and vectorizer."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def preprocess_input(text, vectorizer):
    """Preprocess the input text using the loaded vectorizer."""
    return vectorizer.transform([text])

def predict(model, vectorizer, input_text):
    """Make a prediction using the loaded model and vectorizer."""
    processed_input = preprocess_input(input_text, vectorizer)
    prediction = model.predict(processed_input)
    return prediction

# Example usage
model_path = 'stacking_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'
input_text = "i dont know"

model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
prediction = predict(model, vectorizer, input_text)
print("Prediction:", prediction)