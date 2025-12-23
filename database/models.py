from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class FileMetadataCreate(BaseModel):
    filename: str
    file_type: str
    file_path: str
    file_size: int
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None


class FileMetadataResponse(FileMetadataCreate):
    id: int
    uploaded_at: datetime

    class Config:
        from_attributes = True


class PredictionCreate(BaseModel):
    file_id: int
    model_name: str
    prediction_result: Dict[str, Any]
    confidence_score: Optional[float] = None
    processing_time: float


class PredictionResponse(PredictionCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True