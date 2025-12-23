import numpy as np
import cv2
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from utils.predictor import load_json_weights
weights_path = '../services/utils/weight.json'
# Load pre-trained ResNet50 model for feature extraction
def load_resnet50():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final classification layer
    model.eval()
    return model

model_resnet = load_resnet50()
def extract_resnet_features(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model_resnet(image)
    return features.view(-1).numpy()

def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    return lbp

def ensemble_predict(X_features, X_images, cnn_model, svm_model, dt_model, rf_model):
    cnn_pred = cnn_model.predict(X_images)
    svm_pred = svm_model.predict_proba(X_features)
    dt_pred = dt_model.predict_proba(X_features)
    dt_pred = np.mean(np.array(dt_pred), axis=0)
    rf_pred = rf_model.predict_proba(X_features)
    
     # Weighted ensemble
    weights = load_json_weights(weights_path)
    
    weighted_predictions = (
        weights['cnn'] * cnn_pred +
        weights['decision_tree'] * dt_pred +
        weights['random_forest'] * rf_pred +
        weights['svm'] * svm_pred
    )
    weighted_confidence = (round(np.max(weighted_predictions), 2) * 100)
    final_label = int(np.round(np.mean(weighted_predictions)))
    return final_label, weighted_confidence