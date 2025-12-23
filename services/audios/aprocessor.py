import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import joblib
from tensorflow.keras.models import load_model

from utils.predictor import load_json_weights
weights_path = '../services/utils/weight.json'
# Load models once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)


svm_model_path = '../models/audio/svm_model.joblib'
# decision_tree_model_path = '../models/audio/dt_model.joblib'
random_forest_model_path = '../models/audio/rf_model.joblib'
cnn_path = '../models/audio/cnn_model.h5'

# Load ensemble model components
svm_model = joblib.load(svm_model_path)
# dt_model = joblib.load(decision_tree_model_path)
rf_model = joblib.load(random_forest_model_path)
cnn_model = load_model(cnn_path)

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = wav2vec_model(**inputs)

        feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return feat
    except Exception as e:
        print("Error extracting features with Wav2Vec2:", e)
        return None

def predict_audio_deepfake(file_path):
    features = extract_features(file_path)
    if features is None:
        return None

    # Predict probabilities with individual models
    prob_svm = svm_model.predict_proba(features)[0]
    print("SVM probabilities:", prob_svm)
    # prob_dt = dt_model.predict_proba(features)[0]
    # print("Decision Tree probabilities:", prob_dt)
    # prob_rf = rf_model.predict_proba(features)[0]
    # print("Random Forest probabilities:", prob_rf)

    # CNN model expects 3D input
    features_cnn = features.reshape(-1, 24, 32, 1)
    prob_cnn = cnn_model.predict(features_cnn, verbose=0)[0]
    print("CNN probabilities:", prob_cnn)

    # Weighted ensemble
    weights = load_json_weights(weights_path)
    
    weighted_predictions = (
        0.5 * prob_cnn +
        # weights['decision_tree'] * prob_dt +
        # weights['random_forest'] * prob_rf +
        0.5 * prob_svm
    )

    predicted_class = np.argmax(weighted_predictions)
    confidence_score = np.max(weighted_predictions)
    
    return predicted_class, confidence_score
