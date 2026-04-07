"""
Drop-in replacement for camera.py using a standard webcam via OpenCV.

Provides the same interface as DepthCamera so vision.py works without
changes. Depth is not available — get_distance() always returns a fixed
dummy value so the transformation matrix can still be computed.

Usage: swap the import in main.py from camera to camera_webcam.
"""

import cv2
import numpy as np


class FakeDepthFrame:
    """
    Mimics the pyrealsense2 depth_frame interface.
    Always returns a fixed depth value since a webcam has no depth sensor.
    """
    FIXED_DEPTH = 0.6  # metres — reasonable arm's-length distance

    def get_distance(self, x: int, y: int) -> float:
        return self.FIXED_DEPTH


class DepthCamera:
    """
    Webcam wrapper with the same interface as the RealSense DepthCamera.

    Args:
        device_index: OpenCV camera index (0 = default webcam).
    """

    def __init__(self, device_index: int = 0) -> None:
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {device_index}")
        self._depth_frame = FakeDepthFrame()

    def get_frame(self) -> tuple[bool, np.ndarray | None, FakeDepthFrame | None]:
        """
        Grab the next webcam frame.

        Returns:
            (success, color_image_bgr, fake_depth_frame)
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None
        return True, frame, self._depth_frame

    def release(self) -> None:
        """Release the webcam."""
        self.cap.release()

    def deproject(self, x: int, y: int, depth: float) -> tuple[float, float, float]:
        """
        Fake deprojection for webcam — no real intrinsics available.

        Returns a rough estimate using a pinhole model with assumed FOV.
        X and Y are scaled by depth, Z is depth itself.
        """
        # Approximate focal length for a typical 60° FOV webcam at 640px width
        fx = fy = 600.0
        cx, cy = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2, \
                 self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        return X, Y, depth