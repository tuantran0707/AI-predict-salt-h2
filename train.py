"""
train.py
--------
"Train" few-shot trên 5 mẫu muối: trích feature MobileNetV2 cho từng ảnh
trong thư mục images/ và lưu thành prototypes.npz.

Có làm augmentation nhẹ (flip + thay đổi sáng) để mỗi ảnh gốc tạo ra vài
biến thể, giúp prototype bền hơn với điều kiện ánh sáng trên tàu biển.
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from detect_salt import FeatureExtractor, save_prototypes

IMAGES_DIR = "images"
OUT_PATH = "prototypes.npz"


def augment(bgr: np.ndarray) -> list[np.ndarray]:
    """Sinh thêm biến thể đơn giản cho ảnh gốc."""
    out = [bgr]

    # Lật ngang
    out.append(cv2.flip(bgr, 1))

    # Thay đổi độ sáng
    for gamma in (0.7, 1.3):
        lut = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        ).astype("uint8")
        out.append(cv2.LUT(bgr, lut))

    # Thay đổi cân bằng trắng nhẹ (mô phỏng đèn vàng/đèn trắng)
    warm = bgr.copy().astype(np.int16)
    warm[..., 0] = np.clip(warm[..., 0] - 15, 0, 255)   # bớt blue
    warm[..., 2] = np.clip(warm[..., 2] + 15, 0, 255)   # tăng red
    out.append(warm.astype(np.uint8))

    return out


def main() -> None:
    paths = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMAGES_DIR, "*.png"))
        + glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
    )
    if not paths:
        raise SystemExit(f"Không có ảnh trong '{IMAGES_DIR}/'.")

    print(f"[i] Tìm thấy {len(paths)} ảnh mẫu.")
    print("[i] Tải MobileNetV2 (lần đầu sẽ download ~14MB weights)...")
    extractor = FeatureExtractor()

    embeddings: list[np.ndarray] = []
    names: list[str] = []

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [!] Bỏ qua {p} (không đọc được).")
            continue

        variants = augment(img)
        # Lấy trung bình embedding của các biến thể -> 1 prototype/ảnh
        vecs = np.stack([extractor.embed(v) for v in variants])
        proto = vecs.mean(axis=0)
        proto /= (np.linalg.norm(proto) + 1e-9)

        embeddings.append(proto)
        names.append(os.path.basename(p))
        print(f"  [+] {os.path.basename(p)}: OK ({len(variants)} biến thể)")

    prototypes = np.stack(embeddings).astype(np.float32)
    save_prototypes(OUT_PATH, prototypes, names)
    print(f"[✓] Lưu {prototypes.shape[0]} prototype vào {OUT_PATH} "
          f"(shape={prototypes.shape})")


if __name__ == "__main__":
    main()
