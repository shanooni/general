from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
from images.idetector import predict_image as ipredictor
from audios.aprocessor import audio_predict as apredictor
from audios.audio_processor import MultiLayerAudioFeatureExtractor
import os
import shutil
import mimetypes
import time
from typing import List
# from .database.database import get_db, FileMetadata, Prediction
# from .database.models import FileMetadataCreate, FileMetadataResponse, PredictionCreate, PredictionResponse
# from .services.utils.mime_detector import determine_file_type
# from .services.utils.upload_save import save_upload_file

extractor = MultiLayerAudioFeatureExtractor(
    model_name="facebook/wav2vec2-base-960h",
    layer_indices=[3, 6, 9, 12]
)
app = FastAPI(title = "Deepfake Detection In Multimedia Using Machine Learning ")

origins = [
    "http://localhost:4200",
    "http://localhost:5040",
    "http://localhost:5040/predict",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

label_map = {0: "real", 1: "fake"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Deep fake detection application!"}


@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/v1/info")
async def info():
    return {"app": "Deepfake Detect Application", "version": "1.0.0", "description": "A simple FastAPI application for demonstration of deepfake detection solution."}

@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file for deepfake detection.
    """
    
    if file.filename.endswith(('.jpeg', '.jpg', '.png')):
        try:
            # Read the uploaded file bytes
            file_bytes = await file.read()
            
            # # Convert bytes to PIL Image
            # image = Image.open(io.BytesIO(file_bytes))
            # Pass the PIL Image to the prediction function
            prediction, confidence_score = ipredictor(file_bytes)
            prediction = label_map[prediction]
            confidence = int(round(max(confidence_score), 2) * 100)
            result = {
                "image_name": file.filename,
                "prediction": prediction,
                "confidence": confidence
            }
            return result
        except Exception as e:
            return {"error": str(e)}
    
    elif file.filename.endswith(('.wav', '.mp3', '.flac')):
        try:
            audio_bytes = await file.read()

            print(f"This is an audio file")
            prediction = apredictor(audio_bytes)
            result = {
                "audio_name": file.filename,
                "prediction class": prediction
            }
            return result
        except Exception as e:
            return {"error": str(e)}
    # elif file.filename.endswith(('.mp4', '.avi', '.mov')):
    #     try:
    #         print(f"This is a video file")
    #         # Read the uploaded file bytes
    #         file_bytes = await file.read()
    #         prediction, confidence_score =  vpredictor(file_bytes)
    #         result = {
    #             "video_name": file.filename,
    #             "prediction": "real" if prediction == 1 else "fake",
    #             "confidence": confidence_score
    #         }
    #         return result
    #     except Exception as e:
    #         return {"error": str(e)}

# @app.post("/api/v2/upload", response_model=FileMetadataResponse)
# async def upload_file_v2(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     """
#     Endpoint to upload a file for deepfake detection.
#     """
#     print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
#     file_type = determine_file_type(file.content_type)
    
#     # save file and get path and size
#     file_path, file_size = save_upload_file(file, file_type)

#     # Create FileMetadata object
#     file_metadata = FileMetadataCreate(
#         filename=file.filename,
#         file_path=file_path,
#         file_size=file_size,
#         mime_type=file.content_type
#     )
#     db.add(file_metadata)
#     db.commit()
#     db.refresh(file_metadata)
    
#     return file_metadata

# @app.post("/predict/{file_id}", response_model=PredictionResponse)
# async def predict(
#     file_id: int,
#     model_name: str = "default_model",
#     db: Session = Depends(get_db)
# ):
#     """Run prediction on an uploaded file"""
    
#     # Get file metadata
#     file_metadata = db.query(FileMetadata).filter(FileMetadata.id == file_id).first()
#     if not file_metadata:
#         raise HTTPException(status_code=404, detail="File not found")
    
#     # Run prediction
#     start_time = time.time()
#     prediction_result = ""
#     processing_time = time.time() - start_time
    
#     # Extract confidence score if available
#     confidence_score = prediction_result.get("confidence")
    
    # # Save prediction
    # prediction = Prediction(
    #     file_id=file_id,
    #     model_name=model_name,
    #     prediction_result=prediction_result,
    #     confidence_score=confidence_score,
    #     processing_time=processing_time
    # )
    
    # db.add(prediction)
    # db.commit()
    # db.refresh(prediction)
    
    # return prediction


# @app.post("/api/v2/upload-and-predict/")
# async def upload_and_predict(
#     file: UploadFile = File(...),
#     model_name: str = "default_model",
#     db: Session = Depends(get_db)
# ):
#     """Upload a file and immediately run prediction"""
    
#     # Upload file
#     mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
#     file_type = determine_file_type(mime_type)
#     file_path, file_size = save_upload_file(file, file_type)
    
#     file_metadata = FileMetadata(
#         filename=file.filename,
#         file_type=file_type,
#         file_path=file_path,
#         file_size=file_size,
#         mime_type=mime_type
#     )
    
#     db.add(file_metadata)
#     db.commit()
#     db.refresh(file_metadata)
    
#     # Run prediction
#     start_time = time.time()
#     prediction_result = ""
#     processing_time = time.time() - start_time
    
#     confidence_score = prediction_result.get("confidence")
    
#     prediction = Prediction(
#         file_id=file_metadata.id,
#         model_name=model_name,
#         prediction_result=prediction_result,
#         confidence_score=confidence_score,
#         processing_time=processing_time
#     )
    
#     db.add(prediction)
#     db.commit()
#     db.refresh(prediction)
    
#     return {
#         "file_metadata": FileMetadataResponse.from_orm(file_metadata),
#         "prediction": PredictionResponse.from_orm(prediction)
#     }


# @app.get("/api/v2/files/", response_model=List[FileMetadataResponse])
# async def list_files(
#     skip: int = 0,
#     limit: int = 100,
#     db: Session = Depends(get_db)
# ):
#     """List all uploaded files"""
#     files = db.query(FileMetadata).offset(skip).limit(limit).all()
#     return files


# @app.get("/api/v2/files/{file_id}/predictions", response_model=List[PredictionResponse])
# async def get_file_predictions(
#     file_id: int,
#     db: Session = Depends(get_db)
# ):
#     """Get all predictions for a specific file"""
#     predictions = db.query(Prediction).filter(Prediction.file_id == file_id).all()
#     return predictions


# @app.get("/api/v2/predictions/", response_model=List[PredictionResponse])
# async def list_predictions(
#     skip: int = 0,
#     limit: int = 100,
#     db: Session = Depends(get_db)
# ):
#     """List all predictions"""
#     predictions = db.query(Prediction).offset(skip).limit(limit).all()
#     return predictions


# @app.delete("/api/v2/files/{file_id}")
# async def delete_file(file_id: int, db: Session = Depends(get_db)):
#     """Delete a file and its predictions"""
#     file_metadata = db.query(FileMetadata).filter(FileMetadata.id == file_id).first()
#     if not file_metadata:
#         raise HTTPException(status_code=404, detail="File not found")
    
#     # Delete physical file
#     if os.path.exists(file_metadata.file_path):
#         os.remove(file_metadata.file_path)
    
#     # Delete predictions
#     db.query(Prediction).filter(Prediction.file_id == file_id).delete()
    
#     # Delete metadata
#     db.delete(file_metadata)
#     db.commit()
    
#     return {"message": "File and predictions deleted successfully"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5040)