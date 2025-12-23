import numpy as np
import cv2
from fastapi.responses import JSONResponse

from .helper import preprocess_frame_array, extract_frames_from_video_bytes
from utils.predictor import load_ml_model
from .processor import extract_resnet_features,extract_lbp_features,ensemble_predict

cnn_model_path = "../models/videos/cnn_v2.h5"
svm_model_path = "../models/videos/svm_model.joblib"
dt_model_path = "../models/videos/dt_model.joblib"
rf_model_path = "../models/videos/rf_model.joblib"

cnn_model = load_ml_model(cnn_model_path)
svm_model = load_ml_model(svm_model_path)
dt_model = load_ml_model(dt_model_path)
rf_model = load_ml_model(rf_model_path)

def predict(contents):
    frames = extract_frames_from_video_bytes(contents)
    if not frames:
        return JSONResponse(status_code=400, content={"message": "Could not extract frames from video."})
    
    X_features, X_frames = [], []
    for frame in frames:
        deep_features = extract_resnet_features(frame)
        lbp_features = extract_lbp_features(frame)
        combined_features = np.hstack((deep_features, lbp_features))
        X_features.append(combined_features)
        X_frames.append(cv2.resize(frame, (224, 224)))

    X_features = np.array(X_features)
    X_frames = preprocess_frame_array(X_frames)
    preds, confidence = ensemble_predict(X_features, X_frames, cnn_model, svm_model, dt_model, rf_model)
    return preds, confidence