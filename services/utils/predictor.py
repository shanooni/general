import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

weights = './utils/weight.json'

def load_ml_model(model_path):
    """
    Load a machine learning model from the specified path.
    
    Args:
        model_path (str): Path to the model file.
    
    Returns:
        model: Loaded machine learning model.
    """
    try:
        if model_path.endswith('.h5'):
            # Load Keras model
            model = load_model(model_path)
        elif model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None
    
def load_json_weights(file_path):
    """
    Load weights from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing weights.
    
    Returns:
        dict: Dictionary containing weights for different models.
    """
    import json
    try:
        with open(file_path, 'r') as f:
            weights = json.load(f)
        return weights
    except Exception as e:
        print(f"Error loading weights from {file_path}: {e}")
        return None

def predict(cnn_feature, ml_feature, svm_model_path, 
            log_reg_model_path, random_forest_model_path, 
            cnn_path, weights=weights):
    """
    Predict the class of the input features using a CNN model and a machine learning model.
    
    Args:
        cnn_feature (np.ndarray): Feature vector from the CNN model.
        ml_feature (np.ndarray): Feature vector from the machine learning model.
    
    Returns:
        int: Predicted class label.
    """
    # Assuming cnn_model and ml_model are defined and loaded elsewhere
    svm_model = load_ml_model(svm_model_path) 
    log_reg_model = load_ml_model(log_reg_model_path)
    random_forest_model = load_ml_model(random_forest_model_path)
    cnn_model = load_ml_model(cnn_path)
    
    
    cnn_prediction = cnn_model.predict(cnn_feature)
    print(f"CNN Prediction: {cnn_prediction}")
    log_reg_model_proba = log_reg_model.predict_proba(ml_feature)
    print(f"Logistic Regression Model Probabilities: {log_reg_model_proba}")
    rf_model_proba = random_forest_model.predict_proba(ml_feature)
    print(f"Random Forest Model Probabilities: {rf_model_proba}")  
    svm_model_proba = svm_model.predict_proba(ml_feature)
    print(f"SVM Model Probabilities: {svm_model_proba}")
    
    weights = load_json_weights(weights)
    
    weighted_predictions = (
        weights['cnn'] * cnn_prediction +
        weights['log_reg'] * log_reg_model_proba +
        weights['random_forest'] * rf_model_proba +
        weights['svm'] * svm_model_proba
    )
    print(f"Weighted Predictions: {weighted_predictions}")
    return np.argmax(weighted_predictions, axis=1), np.max(weighted_predictions, axis=1)
