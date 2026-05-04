"""
run_camera.py
-------------
Chạy camera CSI của Raspberry Pi 4 và phát hiện muối bám trên thùng acquy.

Trên Pi 4 (Raspberry Pi OS Bookworm 64-bit) ưu tiên dùng picamera2 + libcamera.
Nếu không có picamera2 (vd. chạy thử trên laptop), tự fallback sang OpenCV
VideoCapture (webcam USB).

Phím tắt:
    q  - thoát
    s  - lưu snapshot vào snapshots/
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np

from detect_salt import SaltDetector

WINDOW = "Salt Detector - Battery"
SNAPSHOT_DIR = "snapshots"


# ---------------------------------------------------------------------------
# Camera abstraction
# ---------------------------------------------------------------------------
class CsiCamera:
    """Picamera2 (CSI) với fallback sang OpenCV VideoCapture."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self._picam = None
        self._cap = None
        self._init_camera()

    def _init_camera(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
            picam = Picamera2()
            config = picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            time.sleep(1.0)  # AWB/AE ổn định
            self._picam = picam
            print("[i] Dùng Picamera2 (CSI).")
            return
        except Exception as e:
            print(f"[!] Không khởi tạo được Picamera2 ({e}). "
                  f"Chuyển sang OpenCV VideoCapture.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Không mở được camera nào.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._picam is not None:
            rgb = self._picam.capture_array()  # RGB
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:
                pass
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# UI overlay
# ---------------------------------------------------------------------------
def draw_overlay(frame: np.ndarray, result: dict, fps: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    has_salt = result["has_salt"]
    conf = result["confidence"]
    color = (0, 0, 255) if has_salt else (0, 200, 0)
    label = "CO MUOI BAM" if has_salt else "SACH"

    # Khung viền cảnh báo
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 6)

    # Panel thông tin
    panel = np.zeros((110, w, 3), dtype=np.uint8)
    cv2.putText(panel, f"{label}  ({conf*100:.1f}%)",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(panel,
                f"AI sim: {result['ai_score']:.2f} | "
                f"CV mask: {result['cv_score']*100:.1f}% | "
                f"match: {result['best_match']} | FPS: {fps:.1f}",
                (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)

    return np.vstack([panel, out])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print("[i] Tải mô hình & prototypes...")
    detector = SaltDetector(prototypes_path="prototypes.npz")

    cam = CsiCamera(width=1280, height=720)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # Suy luận mỗi N frame để giữ FPS mượt trên Pi
    INFER_EVERY = 5
    last_result = {"has_salt": False, "confidence": 0.0, "ai_score": 0.0,
                   "cv_score": 0.0, "best_match": "-"}

    frame_idx = 0
    t0 = time.time()
    fps = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            if frame_idx % INFER_EVERY == 0:
                last_result = detector.predict(frame)

            # FPS
            frame_idx += 1
            if frame_idx % 10 == 0:
                t1 = time.time()
                fps = 10.0 / (t1 - t0)
                t0 = t1

            view = draw_overlay(frame, last_result, fps)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                tag = "salt" if last_result["has_salt"] else "clean"
                fn = os.path.join(SNAPSHOT_DIR, f"{ts}_{tag}.jpg")
                cv2.imwrite(fn, frame)
                print(f"[✓] Snapshot: {fn}")

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
