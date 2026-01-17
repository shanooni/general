import numpy as np
from scipy.special import expit, softmax

class ModelClassifierWrapper:
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        """Always returns an array, never None"""
        try:
            return self.model.predict(X)
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return a default prediction instead of None
            return np.array([0])  # Default to class 0
    
    def predict_proba(self, X):
        """Always returns probability array, never None"""
        try:
            # If model has predict_proba, use it
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            
            # If model has decision_function (like RidgeClassifier), convert to probabilities
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X)
                
                # Binary classification
                if scores.ndim == 1 or scores.shape[1] == 1:
                    if scores.ndim == 2:
                        scores = scores.ravel()
                    proba_positive = expit(scores)
                    return np.vstack([1 - proba_positive, proba_positive]).T
                
                # Multi-class classification
                else:
                    return softmax(scores, axis=1)
            
            # Fallback: return uniform probabilities
            else:
                n_samples = X.shape[0]
                n_classes = len(np.unique(self.model.classes_)) if hasattr(self.model, 'classes_') else 2
                return np.full((n_samples, n_classes), 1.0 / n_classes)
        
        except Exception as e:
            print(f"Probability error: {e}")
            # Return default probabilities [0.5, 0.5] for binary
            return np.array([[0.5, 0.5]])