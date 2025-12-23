import mimetypes


def determine_file_type(mime_type: str) -> str:
    """Determine file type from MIME type"""
    if mime_type.startswith("image/"):
        return "image"
    elif mime_type.startswith("audio/"):
        return "audio"
    elif mime_type.startswith("video/"):
        return "video"
    else:
        return "other"
