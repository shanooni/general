# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

class Config:
    FACE_MESH_STATIC = True
    MAX_FACES = 1
    REFINE_LANDMARKS = True   # Gives 478 landmarks instead of 468
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
    LABELS = ["real", "fake"]  # Folder names representing classes