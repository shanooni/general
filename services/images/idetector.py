import joblib
from .mediapipe import FaceMeshExtractor as facial_extractor


LSVM_MODEL_PATH = "/Users/shanoonissaka/Documents/school/thesis-project/code/general/models/images/calibrated_lsvc.joblib"
RIDGE_MODEL_PATH = "/Users/shanoonissaka/Documents/school/thesis-project/code/general/models/images/calibrated_ridge.joblib"    

lsvm_model = joblib.load(LSVM_MODEL_PATH)
ridge_model = joblib.load(RIDGE_MODEL_PATH)

facial_extractor = facial_extractor()

def predict_image(file):
    """
    Predicts the class of the given image using pre-trained models.
    
    Args:
        file (File): The input file to be classified.
        
    Returns:
        str: The predicted class label.
    """
    # Process the image to get features
    
    features = facial_extractor.extract_from_request(file_bytes=file)
    
    if features is None:
        return {"error": "Face not detected"}
    
    prediction = lsvm_model.predict([features])[0]
    probability = lsvm_model.predict_proba([features])[0] 

    # Use the predictor function to classify the image

    return prediction, probability