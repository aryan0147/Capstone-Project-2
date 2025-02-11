import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import dagshub
import os

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# Handle preflight (OPTIONS) requests to avoid CORS issues
@app.route("/predict_with_timestamps", methods=["OPTIONS"])
@app.route("/predict", methods=["OPTIONS"])
@app.route("/generate_chart", methods=["OPTIONS"])
@app.route("/generate_wordcloud", methods=["OPTIONS"])
@app.route("/generate_trend_graph", methods=["OPTIONS"])
def handle_options():
    return jsonify({"message": "Preflight OK"}), 200


# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r"\n", " ", comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words("english")) - {
            "not",
            "but",
            "however",
            "no",
            "yet",
        }
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "aryan0147"
    repo_name = "Capstone-Project-2"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
    client = MlflowClient()
    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer


# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer(
    "yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl"
)  # Update paths and versions as needed


@app.route("/")
def home():
    return """<h2>Welcome to our Flask API!</h2>
              <p>Check out our Chrome extension:</p>
              <a href="https://github.com/aryan0147/Capstone-2-frontend" target="_blank">GitHub Repository</a>"""


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item["text"] for item in comments_data]
        timestamps = [item["timestamp"] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Return the response with original comments, predicted sentiments, and timestamps
    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render’s assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
