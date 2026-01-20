from .audio_processor import MultiLayerAudioFeatureExtractor
import numpy as np
import joblib

from .modelwrapper import ModelClassifierWrapper

RIDGE_MODEL_PATH = "/Users/shanoonissaka/Documents/school/thesis-project/code/general/models/audio/ridgeclassifier.joblib"
SCALER_PATH = "/Users/shanoonissaka/Documents/school/thesis-project/code/general/models/audio/scaler.joblib"
ridge_model = joblib.load(RIDGE_MODEL_PATH)

extractor = MultiLayerAudioFeatureExtractor(
    model_name="facebook/wav2vec2-base-960h",
    layer_indices=[3, 6, 9, 12]
)
prediction_map = {0 : 'real' , 1 : 'fake'}
def audio_predict(file):
    try:
        # Extract features
        features = extractor.extract_features_from_file(file)
        
        if features is None:
            return {"error": "Unable to extract features from audio"}
        
        if features.ndim == 1:
            # Reshape to 2D
            features = features.reshape(1, -1)
        
        # Predict
        prediction = ridge_model.predict(features)
        
        pred_value = prediction[0]
        pred = int(pred_value)
        return prediction_map[pred]
    
    except Exception as e:
        print(f"Error in audio_predict: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}