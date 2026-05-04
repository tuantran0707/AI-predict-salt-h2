"""
detect_salt.py
--------------
Core detector for salt / sulfate corrosion on marine battery terminals.

Strategy (few-shot, because we only have a handful of samples):
  1) Use MobileNetV2 (ImageNet pretrained, classification head removed)
     as a frozen feature extractor that outputs a 1280-D vector.
  2) Build per-class prototypes:
        - "salt"  : mean embedding of salty / corroded battery samples
        - "clean" : mean embedding of clean battery samples
  3) At inference time, embed the camera frame and compare to both
     prototype banks via cosine similarity. The class with the higher
     similarity wins; the margin is mapped to a confidence score.
  4) A classical HSV mask (white crystals + cyan/green sulfate +
     yellow/brown crust) provides an independent corroboration signal
     that is fused with the AI score.
"""

from __future__ import annotations

import os
import numpy as np
import cv2


# Lazy import so this module can be imported even without TensorFlow installed.
def _load_tf():
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2, preprocess_input,
    )
    return MobileNetV2, preprocess_input


IMG_SIZE = 224  # MobileNetV2 input size


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Wraps MobileNetV2 (include_top=False) with global average pooling."""

    def __init__(self):
        MobileNetV2, preprocess_input = _load_tf()
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
        """Take a single BGR image (OpenCV) and return an L2-normalized vector."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        x = self.preprocess(rgb.astype(np.float32))
        x = np.expand_dims(x, axis=0)
        feat = self.model.predict(x, verbose=0)[0]
        n = np.linalg.norm(feat) + 1e-9
        return feat / n


# ---------------------------------------------------------------------------
# Prototype I/O
# ---------------------------------------------------------------------------
def save_prototypes(path: str,
                    salt_protos: np.ndarray,
                    clean_protos: np.ndarray,
                    salt_names: list[str],
                    clean_names: list[str]) -> None:
    np.savez(
        path,
        salt_protos=salt_protos,
        clean_protos=clean_protos,
        salt_names=np.array(salt_names),
        clean_names=np.array(clean_names),
    )


def load_prototypes(path: str):
    data = np.load(path, allow_pickle=True)
    return (
        data["salt_protos"],
        data["clean_protos"],
        list(data["salt_names"]),
        list(data["clean_names"]),
    )


# ---------------------------------------------------------------------------
# Classical CV signal: ratio of pixels that look like salt / sulfate
# ---------------------------------------------------------------------------
def salt_color_ratio(bgr_image: np.ndarray) -> float:
    """
    Returns a value in [0, 1]: fraction of pixels whose color matches the
    typical look of salt or battery sulfate corrosion. Combines:
      - White / light grey crystals  (low S, high V)
      - Cyan-green copper sulfate    (battery terminal corrosion)
      - Yellow / brown sulfate crust
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    white_mask = (S < 60) & (V > 150)
    cyan_mask = (H > 70) & (H < 110) & (S > 40) & (V > 120)
    yellow_mask = (H > 15) & (H < 35) & (S > 40) & (V > 120)

    salt_mask = white_mask | cyan_mask | yellow_mask
    return float(salt_mask.mean())


# ---------------------------------------------------------------------------
# Combined detector
# ---------------------------------------------------------------------------
class SaltDetector:
    """
    Two-class nearest-prototype classifier with CV-based corroboration.

    Parameters
    ----------
    prototypes_path : path to the .npz file produced by train.py
    margin_threshold : minimum (salt_sim - clean_sim) to flag salt purely
                       from the AI signal
    cv_threshold : minimum HSV mask ratio that on its own raises suspicion
    fusion_weight_ai : weight of the AI signal in the fused confidence
    """

    def __init__(self,
                 prototypes_path: str = "prototypes.npz",
                 margin_threshold: float = 0.03,
                 cv_threshold: float = 0.08,
                 fusion_weight_ai: float = 0.7):
        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(
                f"{prototypes_path} not found. Run: python train.py"
            )
        (self.salt_protos,
         self.clean_protos,
         self.salt_names,
         self.clean_names) = load_prototypes(prototypes_path)

        if len(self.salt_protos) == 0:
            raise ValueError("No 'salt' prototypes found.")
        if len(self.clean_protos) == 0:
            raise ValueError("No 'clean' prototypes found.")

        self.extractor = FeatureExtractor()
        self.margin_threshold = margin_threshold
        self.cv_threshold = cv_threshold
        self.w_ai = fusion_weight_ai

    def predict(self, bgr_image: np.ndarray) -> dict:
        emb = self.extractor.embed(bgr_image)

        salt_sims = self.salt_protos @ emb        # cosine (vectors are L2-normed)
        clean_sims = self.clean_protos @ emb

        salt_sim = float(salt_sims.max())
        clean_sim = float(clean_sims.max())
        margin = salt_sim - clean_sim

        best_salt = self.salt_names[int(salt_sims.argmax())]
        best_clean = self.clean_names[int(clean_sims.argmax())]

        cv_score_raw = salt_color_ratio(bgr_image)
        cv_score = min(cv_score_raw / 0.25, 1.0)   # 0..1

        # AI sub-score in 0..1 (clip the typical margin range -0.1..+0.2)
        ai_score = float(np.clip((margin + 0.1) / 0.3, 0.0, 1.0))
        fused = self.w_ai * ai_score + (1 - self.w_ai) * cv_score

        has_salt = (margin >= self.margin_threshold) or \
                   (cv_score_raw >= self.cv_threshold and margin >= -0.02)

        return {
            "has_salt": bool(has_salt),
            "confidence": float(np.clip(fused, 0.0, 1.0)),
            "salt_sim": salt_sim,
            "clean_sim": clean_sim,
            "margin": margin,
            "cv_ratio": cv_score_raw,
            "best_salt_match": best_salt,
            "best_clean_match": best_clean,
        }
