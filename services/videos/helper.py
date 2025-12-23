
import cv2
import tempfile
import os
import numpy as np

def extract_frames_from_video_bytes(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode="wb") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    return extract_frames(tmp_path)

def extract_frames(video_path, num_frames=15):
    if not os.path.exists(video_path):
        print(f"Error: File path does not exist - {video_path}")
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Unable to read {video_path}. Skipping.")
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames or total_frames == 0:
        print(f"Warning: Video too short or empty - total frames: {total_frames}")
        return []
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
    cap.release()
    return frames

def preprocess_frame_array(frame_array):
    frame_array = np.array(frame_array)
    frame_array = frame_array.astype('float32') / 255.0
    return frame_array