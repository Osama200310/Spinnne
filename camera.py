# Camera abstraction extracted from main.py
from __future__ import annotations

import logging
from typing import Optional

# Try to import picamera2; fall back internally to OpenCV webcam if unavailable
try:
    from picamera2 import Picamera2  # type: ignore
    PICAMERA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PICAMERA_AVAILABLE = False

LOGGER = logging.getLogger("timelapse.camera")


class Camera:
    """Abstraction over Picamera2 or OpenCV VideoCapture.

    Use as a context manager to ensure resources are released:
        with Camera() as cam:
            cam.capture_file("out.jpg")

    New API:
        - capture_array() -> np.ndarray (BGR) for in-memory processing
    """

    def __init__(self, use_picam: bool = True):
        # Honor caller preference but also check actual availability
        self.use_picam: bool = bool(use_picam and PICAMERA_AVAILABLE)
        self.picam2 = None
        self.cv_cap = None

        if self.use_picam:
            self.picam2 = Picamera2()
            camera_config = self.picam2.create_preview_configuration()
            self.picam2.configure(camera_config)
            self.picam2.start()
            LOGGER.info("Using Picamera2")
        else:
            # Use OpenCV VideoCapture for laptop/webcam
            try:
                import cv2  # type: ignore
                self.cv_cap = cv2.VideoCapture(0)
                if not self.cv_cap.isOpened():
                    raise RuntimeError("Cannot open webcam (index 0)")
                # Optional: set resolution
                self.cv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                LOGGER.info("Using OpenCV webcam (index 0)")
            except Exception as e:  # pragma: no cover - hardware dependent
                raise RuntimeError(f"Failed to initialize webcam: {e}")

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def capture_file(self, filename: str) -> None:
        """Capture an image and save to filename"""
        if self.use_picam:
            assert self.picam2 is not None
            self.picam2.capture_file(filename)
        else:
            import cv2  # type: ignore
            assert self.cv_cap is not None
            ret, frame = self.cv_cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from webcam")
            cv2.imwrite(filename, frame)

    def set_controls(self, controls: dict) -> None:
        """Set camera controls (Picamera2 only; ignored for webcam)"""
        if self.use_picam:
            assert self.picam2 is not None
            self.picam2.set_controls(controls)
        else:
            # Webcam controls are limited; log but ignore
            if any(k in controls for k in ("ExposureTime", "AnalogueGain", "FrameDurationLimits")):
                LOGGER.debug("Webcam backend: exposure-related controls ignored")

    def capture_array(self):
        """Capture an image and return a BGR numpy array for in-memory processing.
        Requires numpy; uses OpenCV color conversion for Picamera2 RGB->BGR.
        """
        import numpy as np  # type: ignore
        if self.use_picam:
            assert self.picam2 is not None
            try:
                arr = self.picam2.capture_array()
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Failed to capture array from Picamera2: {e}")
            # Picamera2 returns RGB; convert to BGR for OpenCV/Ultralytics consistency
            try:
                import cv2  # type: ignore
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return bgr
            except Exception:
                # If cv2 isn't available, fall back to RGB array
                return arr
        else:
            import cv2  # type: ignore
            assert self.cv_cap is not None
            ret, frame = self.cv_cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from webcam")
            return frame

    def stop(self) -> None:
        """Stop/release the camera"""
        if self.use_picam and self.picam2:
            self.picam2.stop()
        elif self.cv_cap:
            self.cv_cap.release()
