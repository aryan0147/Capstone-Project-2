import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import mlflow
import mlflow.sklearn
import dagshub  
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature

# Logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Error loading parameters: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path).dropna()
        df.fillna("", inplace=True)
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", model_path)
        return model
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)
        logger.debug("TF-IDF vectorizer loaded from %s", vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error("Error loading vectorizer: %s", e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and return classification metrics and confusion matrix."""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug("Model evaluation completed")
        return report, cm
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_file_path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)
        logger.debug("Model info saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model info: %s", e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
    )


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "aryan0147"
repo_name = "Capstone-Project-2"


def main():
    try:
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
        mlflow.set_experiment("dvc-pipeline-runs")

        with mlflow.start_run() as run:
            root_dir = get_root_directory()
            params = load_params(os.path.join(root_dir, "params.yaml"))

            for key, value in params.items():
                mlflow.log_param(key, value)

            model = load_model(os.path.join(root_dir, "stacking_model.pkl"))
            vectorizer = load_vectorizer(os.path.join(root_dir, "tfidf_vectorizer.pkl"))
            test_data = load_data(
                os.path.join(root_dir, "data/interim/test_processed.csv")
            )

            X_test_tfidf = vectorizer.transform(test_data["clean_comment"].values)
            y_test = test_data["category"].values

            input_example = pd.DataFrame(
                X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out() # type: ignore
            )
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            mlflow.sklearn.log_model(
                model,
                "stacking_model",
                signature=signature,
                input_example=input_example,
            )
            save_model_info(run.info.run_id, "stacking_model", "experiment_info.json")
            mlflow.log_artifact(os.path.join(root_dir, "tfidf_vectorizer.pkl"))

            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log overall accuracy
            accuracy = report["accuracy"]
            mlflow.log_metric("test_accuracy", accuracy)

            # Log precision, recall, and F1-score for each class
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics(
                        {
                            f"test_{label}_precision": metrics["precision"],
                            f"test_{label}_recall": metrics["recall"],
                            f"test_{label}_f1-score": metrics["f1-score"],
                        }
                    )

            log_confusion_matrix(cm, "Test Data")

            mlflow.set_tag("model_type", "StackingClassifier")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

    except Exception as e:
        logger.error("Failed to complete model evaluation: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
