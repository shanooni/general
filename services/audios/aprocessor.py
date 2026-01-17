from .audio_processor import MultiLayerAudioFeatureExtractor
import numpy as np
import joblib

from .modelwrapper import ModelClassifierWrapper

RIDGE_MODEL_PATH = "/Users/shanoonissaka/Documents/school/thesis-project/code/general/models/audio/ridgeClassifier.joblib"
ridge_model = joblib.load(RIDGE_MODEL_PATH)

extractor = MultiLayerAudioFeatureExtractor(
    model_name="facebook/wav2vec2-base-960h",
    layer_indices=[3, 6, 9, 12]
)

def audio_predict(file):
    try:
        # Extract features
        features = extractor.extract_features_from_file(file)
        
        if features is None:
            return {"error": "Unable to extract features from audio"}
        
        # Reshape to 2D
        features = features.reshape(1, -1)
        
        # Predict
        prediction = ridge_model.predict(features)
        # probabilities = ridge_model_wrapper.predict_proba(features)
        
        print(f"prediction is {prediction}")
        # Extract values safely
        pred_value = prediction[0]
        if hasattr(pred_value, "item"):
            pred_value = pred_value.item()
        
        print(f"prediction value {pred_value}")
        return pred_value
    
    except Exception as e:
        print(f"Error in audio_predict: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}