"""Camera abstraction for CSI (Picamera2) and USB webcam fallback."""

from __future__ import annotations

import os
import time

import cv2


class CsiCamera:
    """Picamera2 (CSI) with an OpenCV VideoCapture fallback."""

    def __init__(self, width: int = 1280, height: int = 720,
                 prefer_csi: bool = True, swap_rb: bool = False,
                 camera_format: str = "rgb"):
        self.width = width
        self.height = height
        self.swap_rb = swap_rb
        # Pi/libcamera: RGB888 capture_array is already BGR for OpenCV;
        # BGR888 capture_array is RGB and needs COLOR_RGB2BGR.
        self.camera_format = camera_format
        self._picam = None
        self._cap = None
        if prefer_csi and self._try_picamera2():
            return
        self._open_videocapture()

    def _try_picamera2(self) -> bool:
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception as e:
            print(f"[i] Picamera2 not available ({e}). "
                  "Falling back to OpenCV VideoCapture.")
            return False
        try:
            picam = Picamera2()
            fmt = "BGR888" if self.camera_format == "bgr" else "RGB888"
            # video_configuration = wider field of view (preview crops/zooms).
            config = picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": fmt},
                buffer_count=1,
            )
            picam.configure(config)
            picam.start()
            try:
                picam.set_controls({"FrameRate": 30})
            except Exception:
                pass
            time.sleep(1.0)  # let AWB / AE settle
            self._picam = picam
            print(f"[i] Using Picamera2 (CSI), {self.width}x{self.height}, "
                  f"format={fmt}.", flush=True)
            return True
        except Exception as e:
            print(f"[!] Failed to start Picamera2 ({e}). "
                  "Falling back to OpenCV VideoCapture.")
            return False

    def _open_videocapture(self) -> None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            raise RuntimeError("No camera could be opened.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        self._cap = cap
        backend_name = "DirectShow" if backend == cv2.CAP_DSHOW else "default"
        print(f"[i] Using OpenCV VideoCapture(0) ({backend_name}).")

    def _fix_color(self, frame):
        if self.swap_rb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def read(self):
        if self._picam is not None:
            frame = self._picam.capture_array()
            if self.camera_format == "bgr":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return self._fix_color(frame)
        ok, frame = self._cap.read()
        return self._fix_color(frame) if ok else None

    def release(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:
                pass
        if self._cap is not None:
            self._cap.release()

    def warmup(self, frames: int = 5, delay_s: float = 0.1) -> None:
        """Discard a few frames so auto-exposure can settle."""
        for _ in range(frames):
            self.read()
            time.sleep(delay_s)
