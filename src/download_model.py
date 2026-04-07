"""
download_model.py
-----------------
Downloads the MediaPipe HandLandmarker model file required by vision.py.

Run once before starting the system:
    uv run python download_model.py
"""

import urllib.request
from pathlib import Path

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path("hand_landmarker.task")


def download() -> None:
    if MODEL_PATH.exists():
        print(f"Model already present at '{MODEL_PATH}'. Nothing to do.")
        return

    print(f"Downloading HandLandmarker model to '{MODEL_PATH}'...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    download()