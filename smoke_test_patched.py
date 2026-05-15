"""Quick offline accuracy check on the dataset/ images."""
import cv2
import glob
import os

from detect_salt import SaltDetector

d = SaltDetector()
groups = [("salt", sorted(glob.glob("dataset/salt/*.jpg"))),
          ("clean", sorted(glob.glob("dataset/clean/*.jpg")))]
correct = total = 0
for label, paths in groups:
    for p in paths:
        img = cv2.imread(p); if img is None: continue
        r = d.predict(img)
        ok = (label == "salt") == r["has_salt"]
        correct += ok
        total += 1
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {os.path.basename(p):14s} "
              f"true={label:5s}  pred_salt={str(r['has_salt']):5s} "
              f"margin={r['margin']:+.3f} conf={r['confidence']:.2f} "
              f"cv={r['cv_ratio']*100:5.1f}%")
print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
