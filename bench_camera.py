"""Live FPS / detection benchmark on the Pi CSI camera."""
import time
import cv2
from picamera2 import Picamera2

from detect_salt import SaltDetector

d = SaltDetector()
p = Picamera2()
p.configure(p.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}))
p.start()
time.sleep(1)

n = 0
t0 = time.time()
while time.time() - t0 < 8:
    rgb = p.capture_array()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    r = d.predict(bgr)
    n += 1
    if n % 5 == 0:
        print(f"frame {n:3d}  has_salt={r['has_salt']}  "
              f"margin={r['margin']:+.2f}  cv={r['cv_ratio']*100:.1f}%")

dt = time.time() - t0
print(f"\nTOTAL {n} frames in {dt:.1f}s -> {n/dt:.2f} FPS")
p.stop()
