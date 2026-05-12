"""
detect_salt.py
--------------
Salt / sulfate corrosion detector for marine batteries.

Few-shot pipeline that runs everywhere OpenCV runs (no TensorFlow):
  1) MobileNetV2 (ImageNet pretrained, ONNX) is loaded via cv2.dnn.
     Its 1000-D logit vector is used as an image embedding.
  2) Per-class prototypes are pre-computed by train.py:
        - "salt"  : mean embedding of corroded battery samples
        - "clean" : mean embedding of clean battery samples
  3) At inference time the camera frame is embedded once and compared
     against both banks via cosine similarity. The class with the higher
     similarity wins; the margin is mapped to a confidence value.
  4) An independent HSV color mask (white crystals + cyan/green sulfate +
     yellow/brown crust) is fused in to corroborate the AI signal.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_MODEL = os.path.join("model", "mobilenetv2-12.onnx")


# ---------------------------------------------------------------------------
# Feature extractor (OpenCV DNN + MobileNetV2 ONNX)
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """MobileNetV2 ONNX wrapped in cv2.dnn.

    Returns an L2-normalized 1000-D logit vector per image. Logits work
    fine as embeddings for cosine-similarity-based few-shot classification.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at: {model_path}. "
                "Download mobilenetv2-12.onnx from the ONNX model zoo."
            )
        self.net = cv2.dnn.readNetFromONNX(model_path)
        # cv2.dnn picks the best available CPU backend automatically.
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        # NCHW
        x = np.transpose(x, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return x

    def embed(self, bgr_image: np.ndarray) -> np.ndarray:
        blob = self._preprocess(bgr_image)
        self.net.setInput(blob)
        feat = self.net.forward().flatten()
        n = float(np.linalg.norm(feat)) + 1e-9
        return (feat / n).astype(np.float32)


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
    """Returns a value in [0, 1]: fraction of pixels matching salt or
    battery sulfate corrosion colors:

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


def salt_mask(bgr_image: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of pixels likely to be salt / sulfate."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    white_mask = (S < 60) & (V > 150)
    cyan_mask = (H > 70) & (H < 110) & (S > 40) & (V > 120)
    yellow_mask = (H > 15) & (H < 35) & (S > 40) & (V > 120)
    m = (white_mask | cyan_mask | yellow_mask).astype(np.uint8) * 255
    # Clean up noise and merge nearby crystals
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    return m


def salt_boxes(bgr_image: np.ndarray,
               min_area_ratio: float = 0.002,
               max_boxes: int = 8) -> list:
    """Return bounding boxes around suspected salt regions.

    Each box is (x, y, w, h, area_ratio). `area_ratio` is the share of the
    image covered by that blob, useful for ranking and labeling.
    """
    h, w = bgr_image.shape[:2]
    img_area = float(h * w)
    mask = salt_mask(bgr_image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area / img_area < min_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((x, y, bw, bh, area / img_area))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes[:max_boxes]


# ---------------------------------------------------------------------------
# Combined detector
# ---------------------------------------------------------------------------
class SaltDetector:
    """Two-class nearest-prototype classifier with CV-based corroboration.

    Parameters
    ----------
    prototypes_path : .npz produced by train.py
    model_path      : MobileNetV2 ONNX file
    margin_threshold : minimum (salt_sim - clean_sim) to flag salt purely
                       from the AI signal
    cv_threshold : minimum HSV mask ratio that on its own raises suspicion
    fusion_weight_ai : weight of the AI signal in the fused confidence
    """

    def __init__(self,
                 prototypes_path: str = "prototypes.npz",
                 model_path: str = DEFAULT_MODEL,
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

        self.extractor = FeatureExtractor(model_path)
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

        boxes = salt_boxes(bgr_image) if has_salt else []

        return {
            "has_salt": bool(has_salt),
            "confidence": float(np.clip(fused, 0.0, 1.0)),
            "salt_sim": salt_sim,
            "clean_sim": clean_sim,
            "margin": margin,
            "cv_ratio": cv_score_raw,
            "best_salt_match": best_salt,
            "best_clean_match": best_clean,
            "boxes": boxes,
        }
