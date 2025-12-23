import os
import cv2
import mediapipe as mp
import numpy as np
import io
from PIL import Image
from .config import Config

class FaceMeshExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.model = self.mp_face_mesh.FaceMesh(
            static_image_mode=Config.FACE_MESH_STATIC,
            max_num_faces=Config.MAX_FACES,
            refine_landmarks=Config.REFINE_LANDMARKS
        )

    def extract(self, image_path: str):
        """Extract and return flattened (x,y,z) landmark array for a single face."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.model.process(img_rgb)
        if not results.multi_face_landmarks:
            print("No face detected in image:", image_path)
            return None  # No face detected

        # Always use the first face (index 0)
        landmarks = results.multi_face_landmarks[0]

        coords = []
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        return coords
    
    def extract_from_request(self, file_bytes: bytes):
        """
        Extract landmark features from an uploaded image (bytes).
        Suitable for FastAPI: file.file.read()
        """
        # Convert raw bytes → numpy array → OpenCV image
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            try:
                pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                img = np.array(pil_image)[:,:,::-1]  # Convert RGB to BGR for OpenCV
            except Exception as e:
                print("Error reading uploaded image:", e)
                return None
            

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)

        if not results.multi_face_landmarks:
            print("No face detected in uploaded image")
            return None

        landmarks = results.multi_face_landmarks[0]

        coords = []
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        return coords