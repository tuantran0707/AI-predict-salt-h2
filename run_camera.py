"""
run_camera.py
-------------
Live salt / sulfate detection on a marine battery using the Raspberry Pi 4
CSI camera (or any USB webcam on a development machine).

On the Pi we use Picamera2 (libcamera) for the CSI port. On a development
machine without picamera2 the script automatically falls back to OpenCV
VideoCapture (a USB webcam).

Hotkeys (GUI mode):
    q  - quit
    s  - save a snapshot to snapshots/
"""

from __future__ import annotations

import argparse
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
    """Picamera2 (CSI) with an OpenCV VideoCapture fallback."""

    def __init__(self, width: int = 1280, height: int = 720,
                 prefer_csi: bool = True):
        self.width = width
        self.height = height
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
            config = picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            time.sleep(1.0)  # let AWB / AE settle
            self._picam = picam
            print("[i] Using Picamera2 (CSI).")
            return True
        except Exception as e:
            print(f"[!] Failed to start Picamera2 ({e}). "
                  "Falling back to OpenCV VideoCapture.")
            return False

    def _open_videocapture(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No camera could be opened.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        print("[i] Using OpenCV VideoCapture(0).")

    def read(self) -> np.ndarray | None:
        if self._picam is not None:
            rgb = self._picam.capture_array()
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
    label = "SALT DETECTED" if has_salt else "CLEAN"

    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 6)

    panel = np.zeros((110, w, 3), dtype=np.uint8)
    cv2.putText(panel, f"{label}  ({conf*100:.1f}%)",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(panel,
                f"salt_sim: {result['salt_sim']:.2f}  "
                f"clean_sim: {result['clean_sim']:.2f}  "
                f"margin: {result['margin']:+.2f}  "
                f"cv: {result['cv_ratio']*100:.1f}%  "
                f"FPS: {fps:.1f}",
                (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)

    return np.vstack([panel, out])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Salt detector on CSI camera.")
    p.add_argument("--prototypes", default="prototypes.npz")
    p.add_argument("--model", default=os.path.join("model",
                                                   "mobilenetv2-12.onnx"))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--infer-every", type=int, default=5,
                   help="Run inference every N frames (smoother preview).")
    p.add_argument("--headless", action="store_true",
                   help="No GUI window. Print results to stdout, save "
                        "snapshots when salt is detected.")
    p.add_argument("--no-csi", action="store_true",
                   help="Skip Picamera2 and use OpenCV VideoCapture only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print("[i] Loading model and prototypes...")
    detector = SaltDetector(prototypes_path=args.prototypes,
                            model_path=args.model)

    cam = CsiCamera(width=args.width, height=args.height,
                    prefer_csi=not args.no_csi)

    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    last_result = {
        "has_salt": False, "confidence": 0.0,
        "salt_sim": 0.0, "clean_sim": 0.0, "margin": 0.0,
        "cv_ratio": 0.0,
        "best_salt_match": "-", "best_clean_match": "-",
    }
    last_alert_time = 0.0
    ALERT_COOLDOWN_S = 5.0

    frame_idx = 0
    t0 = time.time()
    fps = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            if frame_idx % args.infer_every == 0:
                last_result = detector.predict(frame)

            frame_idx += 1
            if frame_idx % 10 == 0:
                t1 = time.time()
                fps = 10.0 / (t1 - t0)
                t0 = t1

            if args.headless:
                now = time.time()
                if (last_result["has_salt"]
                        and now - last_alert_time > ALERT_COOLDOWN_S):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn = os.path.join(SNAPSHOT_DIR, f"{ts}_salt.jpg")
                    cv2.imwrite(fn, frame)
                    print(f"[ALERT] {ts}  conf={last_result['confidence']:.2f}"
                          f"  margin={last_result['margin']:+.2f}"
                          f"  saved={fn}")
                    last_alert_time = now
                time.sleep(0.01)   # avoid pegging CPU on the Pi
            else:
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
                    print(f"[OK] Snapshot saved: {fn}")
    except KeyboardInterrupt:
        print("\n[i] Interrupted, shutting down.")
    finally:
        cam.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
