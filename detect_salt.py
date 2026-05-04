"""
detect_salt.py
--------------
Lõi nhận diện muối bám trên cọc/ thùng acquy.

Cách tiếp cận (few-shot vì chỉ có 5 mẫu):
  1) Dùng MobileNetV2 pretrained ImageNet (bỏ lớp phân loại) để trích
     feature vector 1280 chiều cho mỗi ảnh.
  2) Lưu 5 vector của 5 mẫu thành "prototype".
  3) Khi suy luận: tính cosine similarity của frame với 5 prototype,
     lấy max => điểm AI.
  4) Kết hợp với điểm CV cổ điển (mask HSV vùng trắng/sáng dạng tinh thể)
     => quyết định cuối cùng.
"""

from __future__ import annotations

import os
import numpy as np
import cv2

# Lazy import TF để file này vẫn import được khi chỉ dùng phần CV
def _load_tf():
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2, preprocess_input,
    )
    return tf, MobileNetV2, preprocess_input


IMG_SIZE = 224  # Đầu vào MobileNetV2


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Bọc MobileNetV2 (include_top=False) + GlobalAveragePooling."""

    def __init__(self):
        tf, MobileNetV2, preprocess_input = _load_tf()
        base = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        base.trainable = False
        self.model = base
        self.preprocess = preprocess_input

    def embed(self, bgr_image: np.ndarray) -> np.ndarray:
        """Nhận 1 ảnh BGR (OpenCV) -> vector đặc trưng đã chuẩn hoá L2."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        x = self.preprocess(rgb.astype(np.float32))
        x = np.expand_dims(x, axis=0)
        feat = self.model.predict(x, verbose=0)[0]
        # Chuẩn hoá L2 để cosine = dot product
        n = np.linalg.norm(feat) + 1e-9
        return feat / n


# ---------------------------------------------------------------------------
# Prototype I/O
# ---------------------------------------------------------------------------
def save_prototypes(path: str, prototypes: np.ndarray, names: list[str]) -> None:
    np.savez(path, prototypes=prototypes, names=np.array(names))


def load_prototypes(path: str) -> tuple[np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    return data["prototypes"], list(data["names"])


# ---------------------------------------------------------------------------
# Tín hiệu CV cổ điển: tỉ lệ pixel "trắng/sáng dạng tinh thể"
# ---------------------------------------------------------------------------
def salt_color_ratio(bgr_image: np.ndarray) -> float:
    """
    Trả về tỉ lệ (0..1) pixel có màu sáng/ngả trắng - đặc trưng của tinh thể
    muối/sulfat bám trên cọc acquy. Kết hợp:
      - HSV: S thấp + V cao (trắng/xám sáng)
      - Hoặc tone xanh nhạt - vàng nhạt của sulfat đồng (CuSO4) thường gặp
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Tinh thể trắng/xám sáng
    white_mask = (S < 60) & (V > 150)

    # Sulfat xanh ngọc (battery corrosion)
    cyan_mask = (H > 70) & (H < 110) & (S > 40) & (V > 120)

    # Sulfat vàng/nâu nhạt
    yellow_mask = (H > 15) & (H < 35) & (S > 40) & (V > 120)

    salt_mask = white_mask | cyan_mask | yellow_mask
    return float(salt_mask.mean())


# ---------------------------------------------------------------------------
# Detector tổng hợp
# ---------------------------------------------------------------------------
class SaltDetector:
    def __init__(self, prototypes_path: str = "prototypes.npz",
                 ai_threshold: float = 0.55,
                 cv_threshold: float = 0.08,
                 fusion_weight_ai: float = 0.7):
        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(
                f"Không tìm thấy {prototypes_path}. Hãy chạy: python train.py"
            )
        self.prototypes, self.names = load_prototypes(prototypes_path)
        self.extractor = FeatureExtractor()
        self.ai_threshold = ai_threshold
        self.cv_threshold = cv_threshold
        self.w_ai = fusion_weight_ai

    def predict(self, bgr_image: np.ndarray) -> dict:
        # Điểm AI (cosine sim cao nhất với 5 prototype)
        emb = self.extractor.embed(bgr_image)
        sims = self.prototypes @ emb            # đã L2-norm => cosine
        ai_score = float(sims.max())
        best_idx = int(sims.argmax())

        # Điểm CV
        cv_score_raw = salt_color_ratio(bgr_image)
        # Đưa cv_score về dải 0..1 (lấy threshold 0.25 là "rất nhiều muối")
        cv_score = min(cv_score_raw / 0.25, 1.0)

        # Hợp nhất
        ai_norm = max(0.0, (ai_score - 0.3) / 0.7)   # 0.3..1 -> 0..1
        fused = self.w_ai * ai_norm + (1 - self.w_ai) * cv_score

        has_salt = (ai_score >= self.ai_threshold) or \
                   (cv_score_raw >= self.cv_threshold and ai_score >= 0.45)

        return {
            "has_salt": bool(has_salt),
            "confidence": float(np.clip(fused, 0.0, 1.0)),
            "ai_score": ai_score,
            "cv_score": cv_score_raw,
            "best_match": self.names[best_idx],
        }
