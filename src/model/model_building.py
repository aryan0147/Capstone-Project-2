import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path).dropna()
        df.fillna('', inplace=True)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with n-grams to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values
        X_train_tfidf = vectorizer.fit_transform(X_train)

        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug(f'TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}')
        return X_train_tfidf, y_train, vectorizer
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

def train_base_models(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    """Train LightGBM and Logistic Regression models."""
    try:
        lightgbm_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            is_unbalance=True,
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators']
        )
        lightgbm_model.fit(X_train, y_train)

        logreg_model = LogisticRegression(
            max_iter=1000, C=params['logreg_C'], penalty=params['logreg_penalty'], solver='liblinear'
        )
        logreg_model.fit(X_train, y_train)

        logger.debug('Base models trained successfully')
        return lightgbm_model, logreg_model
    except Exception as e:
        logger.error('Error training base models: %s', e)
        raise

def train_stacking_model(X_train: np.ndarray, y_train: np.ndarray, base_models: tuple):
    """Train the stacking model with KNN as meta-learner."""
    try:
        stacking_model = StackingClassifier(
            estimators=[
                ('lightgbm', base_models[0]),
                ('logistic_regression', base_models[1])
            ],
            final_estimator=KNeighborsClassifier(n_neighbors=5),
            cv=5
        )
        stacking_model.fit(X_train, y_train)
        logger.debug('Stacking model training completed')
        return stacking_model
    except Exception as e:
        logger.error('Error training stacking model: %s', e)
        raise

def save_model(model, file_path: str):
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving model: %s', e)
        raise

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))['model_building']
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
        X_train_tfidf, y_train, vectorizer = apply_tfidf(train_data, params['max_features'], tuple(params['ngram_range']))
        
        base_models = train_base_models(X_train_tfidf, y_train, params)
        stacking_model = train_stacking_model(X_train_tfidf, y_train, base_models)

        save_model(stacking_model, os.path.join(root_dir, 'stacking_model.pkl'))
        save_model(vectorizer, os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
    except Exception as e:
        logger.error('Failed to complete model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
