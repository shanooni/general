import shutil
from pathlib import Path
from fastapi import UploadFile


# Configure file storage
UPLOAD_DIR = Path("./Users/shanoonissaka/Documents/school/thesis-project/code/general/services/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Create subdirectories for different file types
(UPLOAD_DIR / "images").mkdir(exist_ok=True)
(UPLOAD_DIR / "audio").mkdir(exist_ok=True)
(UPLOAD_DIR / "videos").mkdir(exist_ok=True)

def save_upload_file(upload_file: UploadFile, file_type: str) -> tuple[str, int]:
    """Save uploaded file and return path and size"""
    file_path = UPLOAD_DIR / file_type + "s" / upload_file.filename
    
    # Ensure unique filename
    counter = 1
    while file_path.exists():
        stem = Path(upload_file.filename).stem
        suffix = Path(upload_file.filename).suffix
        file_path = UPLOAD_DIR / file_type + "s" / f"{stem}_{counter}{suffix}"
        counter += 1
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    file_size = file_path.stat().st_size
    return str(file_path), file_size