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

# Legacy fixed ROIs — only used when auto terminal detection fails.
FALLBACK_TERMINAL_ROIS = (
    {"name": "L", "x": 0.01, "y": 0.01, "w": 0.30, "h": 0.28},
    {"name": "R", "x": 0.69, "y": 0.01, "w": 0.30, "h": 0.28},
)
# Kept for backward compatibility; prefer auto detection.
DEFAULT_TERMINAL_ROIS = FALLBACK_TERMINAL_ROIS


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
      - Yellow / amber sulfate crust (warm shipboard lighting shifts hue)
      - Brown / dark sulfate deposits
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    white_mask = (S < 60) & (V > 150)
    cyan_mask = (H > 70) & (H < 110) & (S > 40) & (V > 120)
    # Yellow/amber: widened H for warm lighting (H~40-50 under tungsten lamps)
    yellow_mask = (H > 8) & (H < 55) & (S > 25) & (V > 70)
    # Dark brown / oxidised lead crust
    brown_mask = (H > 5) & (H < 35) & (S > 20) & (V > 45) & (V < 160)

    salt_mask = white_mask | cyan_mask | yellow_mask | brown_mask
    return float(salt_mask.mean())


def terminal_salt_ratio(bgr_image: np.ndarray) -> float:
    """Salt/colour ratio on a terminal-head crop (no broad white/case mask)."""
    return max(_yellow_salt_ratio(bgr_image), _corrosion_salt_ratio(bgr_image))


def _yellow_salt_ratio(bgr_image: np.ndarray) -> float:
    """Yellow/amber sulfate crust (warm lighting, corner terminals)."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    yellow_mask = (H > 6) & (H < 65) & (S > 12) & (V > 55)
    return float(yellow_mask.mean())


def _corrosion_salt_ratio(bgr_image: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    cyan_mask = (H > 70) & (H < 110) & (S > 40) & (V > 120)
    brown_mask = (H > 5) & (H < 35) & (S > 20) & (V > 45) & (V < 160)
    white_mask = (S < 45) & (V > 175)
    salt_mask = cyan_mask | brown_mask | white_mask
    return float(salt_mask.mean())


def salt_mask(bgr_image: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of pixels likely to be salt / sulfate."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    white_mask = (S < 60) & (V > 150)
    cyan_mask = (H > 70) & (H < 110) & (S > 40) & (V > 120)
    yellow_mask = (H > 8) & (H < 55) & (S > 25) & (V > 70)
    brown_mask = (H > 5) & (H < 35) & (S > 20) & (V > 45) & (V < 160)
    m = (white_mask | cyan_mask | yellow_mask | brown_mask).astype(np.uint8) * 255
    # Clean up noise and merge nearby crystals
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    return m


def salt_boxes(bgr_image: np.ndarray,
               min_area_ratio: float = 0.002,
               max_boxes: int = 8,
               roi: dict | None = None) -> list:
    """Return bounding boxes around suspected salt regions.

    Each box is (x, y, w, h, area_ratio). `area_ratio` is the share of the
    image covered by that blob, useful for ranking and labeling.

    If *roi* is given (fractional x/y/w/h), search only inside that crop and
    return boxes in full-frame coordinates.
    """
    offset_x = offset_y = 0
    img = bgr_image
    if roi is not None:
        img, (offset_x, offset_y) = _crop_roi(bgr_image, roi)

    h, w = img.shape[:2]
    img_area = float(h * w)
    mask = salt_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area / img_area < min_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((x + offset_x, y + offset_y, bw, bh, area / img_area))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes[:max_boxes]


def _roi_rect(roi: dict, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    x = int(roi["x"] * frame_w)
    y = int(roi["y"] * frame_h)
    w = int(roi["w"] * frame_w)
    h = int(roi["h"] * frame_h)
    return x, y, w, h


def _crop_roi(bgr_image: np.ndarray,
              roi: dict) -> tuple[np.ndarray, tuple[int, int]]:
    fh, fw = bgr_image.shape[:2]
    x, y, w, h = _roi_rect(roi, fh, fw)
    return bgr_image[y:y + h, x:x + w], (x, y)


def find_terminal_heads(bgr_image: np.ndarray,
                        top_frac: float = 0.42) -> list[dict]:
    """Locate the two battery terminal posts in the top band of the frame.

    Returns a list of dicts with keys: name, x, y, w, h (fractional crop),
    and head_rect (x, y, w, h) in full-frame pixels for drawing.
    """
    fh, fw = bgr_image.shape[:2]
    band_h = max(32, int(fh * top_frac))
    band = bgr_image[0:band_h, :]
    band_area = float(band.shape[0] * band.shape[1])

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    # Terminals rise above the black case — brighter / not deep black plastic.
    mask = ((V > 55) & (S < 130)) | ((V > 90) & (S < 160))
    mask = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict] = []
    for c in contours:
        area = cv2.contourArea(c)
        ratio = area / band_area
        if ratio < 0.0008 or ratio > 0.12:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 8 or bh < 8:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.25 or aspect > 4.0:
            continue
        cx = x + bw / 2.0
        candidates.append({
            "cx": cx, "x": x, "y": y, "w": bw, "h": bh, "area": area,
        })

    # Salt/corrosion blobs also mark terminal positions in the top band.
    salt_band = salt_mask(band)
    s_contours, _ = cv2.findContours(salt_band, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    for c in s_contours:
        area = cv2.contourArea(c)
        ratio = area / band_area
        if ratio < 0.0005 or ratio > 0.08:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 6 or bh < 6:
            continue
        cx = x + bw / 2.0
        candidates.append({
            "cx": cx, "x": x, "y": y, "w": bw, "h": bh,
            "area": area + band_area * 0.001,
        })

    if not candidates:
        return []

    left = [c for c in candidates if c["cx"] < fw * 0.48]
    right = [c for c in candidates if c["cx"] > fw * 0.52]
    poles: list[dict] = []
    for name, group in (("L", left), ("R", right)):
        if not group:
            continue
        best = max(group, key=lambda c: c["area"])
        pad_w = int(best["w"] * 0.65)
        pad_h = int(best["h"] * 0.65)
        x1 = max(0, best["x"] - pad_w)
        y1 = max(0, best["y"] - pad_h)
        x2 = min(fw, best["x"] + best["w"] + pad_w)
        y2 = min(fh, best["y"] + best["h"] + pad_h)
        bw, bh = x2 - x1, y2 - y1
        poles.append({
            "name": name,
            "x": x1 / fw,
            "y": y1 / fh,
            "w": bw / fw,
            "h": bh / fh,
            "head_rect": (x1, y1, bw, bh),
        })
    return poles


def _resolve_terminal_rois(bgr_image: np.ndarray,
                           rois: tuple[dict, ...] | str | None
                           ) -> list[dict]:
    """Auto-detect terminal heads, or fall back to small fixed corner ROIs."""
    if rois is None:
        return []
    if rois == "auto":
        found = find_terminal_heads(bgr_image)
        if found:
            return found
        return [
            {**r, "head_rect": _roi_rect(r, *bgr_image.shape[:2])}
            for r in FALLBACK_TERMINAL_ROIS
        ]
    return [{**r, "head_rect": _roi_rect(r, *bgr_image.shape[:2])}
            for r in rois]


def analyze_poles(bgr_image: np.ndarray,
                  rois: tuple[dict, ...] | str,
                  extractor: FeatureExtractor,
                  salt_protos: np.ndarray,
                  clean_protos: np.ndarray,
                  margin_threshold: float,
                  pole_cv_threshold: float,
                  fusion_weight_ai: float = 0.7,
                  min_area_ratio: float = 0.004) -> list[dict]:
    """Per-terminal-head salt check (auto-located posts, not fixed half-frame)."""
    terminal_rois = _resolve_terminal_rois(bgr_image, rois)
    poles: list[dict] = []
    for roi in terminal_rois:
        crop, (ox, oy) = _crop_roi(bgr_image, roi)
        if crop.size == 0:
            continue

        cv_ratio = terminal_salt_ratio(crop)
        yellow_cv = _yellow_salt_ratio(crop)

        emb = extractor.embed(crop)
        salt_sim = float((salt_protos @ emb).max())
        clean_sim = float((clean_protos @ emb).max())
        margin = salt_sim - clean_sim

        cv_score = min(cv_ratio / 0.15, 1.0)
        ai_score = float(np.clip((margin + 0.30) / 0.60, 0.0, 1.0))
        fused = fusion_weight_ai * ai_score + (1 - fusion_weight_ai) * cv_score

        blob = salt_boxes(crop, min_area_ratio=min_area_ratio, max_boxes=1)
        box = None
        has_blob = False
        if blob:
            x, y, bw, bh, area_ratio = blob[0]
            box = (x + ox, y + oy, bw, bh, area_ratio)
            has_blob = True

        # Yellow crust at corner posts: colour evidence + AI (margin may be weak).
        has_salt = (
            margin >= margin_threshold
            or (margin >= 0.0 and cv_ratio >= pole_cv_threshold)
            or (margin >= 0.0 and has_blob
                and cv_ratio >= pole_cv_threshold * 0.6)
            or (margin >= 0.03 and cv_ratio >= 0.08)
            or (yellow_cv >= 0.07 and margin >= -0.05)
            or (yellow_cv >= 0.10 and cv_ratio >= 0.05)
        )

        head_rect = roi.get("head_rect", _roi_rect(roi, *bgr_image.shape[:2]))
        poles.append({
            "name": roi["name"],
            "has_salt": has_salt,
            "confidence": fused,
            "cv_ratio": cv_ratio,
            "yellow_cv": yellow_cv,
            "margin": margin,
            "salt_sim": salt_sim,
            "clean_sim": clean_sim,
            "box": box,
            "head_rect": head_rect,
        })
    return poles


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
    cv_threshold : minimum HSV mask ratio for full-frame fallback mode
    pole_cv_threshold : per-terminal minimum cv_ratio to flag SALT;
                        default 0.40 (40 %); also fused with AI per pole
    fusion_weight_ai : weight of the AI signal in the fused confidence
    """

    def __init__(self,
                 prototypes_path: str = "prototypes.npz",
                 model_path: str = DEFAULT_MODEL,
                 margin_threshold: float = 0.025,
                 cv_threshold: float = 0.03,
                 pole_cv_threshold: float = 0.10,
                 fusion_weight_ai: float = 0.72,
                 terminal_rois: tuple[dict, ...] | str | None = "auto"):
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
        self.pole_cv_threshold = pole_cv_threshold
        self.w_ai = fusion_weight_ai
        self.terminal_rois = terminal_rois

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
        poles: list[dict] = []
        if self.terminal_rois is not None:
            poles = analyze_poles(
                bgr_image, self.terminal_rois,
                self.extractor, self.salt_protos, self.clean_protos,
                self.margin_threshold, self.pole_cv_threshold, self.w_ai)
            if poles:
                cv_score_raw = max((p["cv_ratio"] for p in poles), default=0.0)

        # Empirical (74 salt + 173 clean prototypes): P95 of salt cv_ratio ~ 0.07
        cv_score = min(cv_score_raw / 0.08, 1.0)   # 0..1

        # AI sub-score in 0..1. Empirical margin distribution on this dataset:
        #   salt  margin ~ N(+0.165, 0.082), P5=+0.053, P95=+0.309
        #   clean margin ~ N(-0.165, 0.077), P5=-0.289, P95=-0.049
        # Mapping the symmetric range [-0.30, +0.30] to [0, 1].
        ai_score = float(np.clip((margin + 0.30) / 0.60, 0.0, 1.0))
        fused = self.w_ai * ai_score + (1 - self.w_ai) * cv_score

        pole_cv_hit = any(p["has_salt"] for p in poles)
        if poles:
            has_salt = pole_cv_hit
            if not has_salt and margin >= 0.06:
                best = max(poles, key=lambda p: p["cv_ratio"])
                if best["cv_ratio"] >= 0.12 and best["margin"] >= -0.03:
                    has_salt = True
                    best["has_salt"] = True
            conf_poles = [p["confidence"] for p in poles if p["has_salt"]]
            if conf_poles:
                fused = max(conf_poles)
            boxes = [p["box"] for p in poles
                     if p["has_salt"] and p["box"] is not None]
        else:
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
            "poles": poles,
            "boxes": boxes,
        }


def jpeg_to_data_uri(jpeg_bytes: bytes) -> str:
    """Data-URI for ThingsBoard image widget: data:image/jpeg;base64,..."""
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


def frame_to_base64(bgr_image: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR frame as a base64 JPEG string."""
    import base64
    ok, buf = cv2.imencode(".jpg", bgr_image,
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def compress_jpeg_bytes(jpeg_bytes: bytes, max_width: int = 480,
                        quality: int = 50,
                        max_bytes: int = 22000) -> bytes:
    """Shrink a JPEG so the MQTT payload stays under ThingsBoard limits (~64 KB)."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)

    out = jpeg_bytes
    for q in (quality, 45, 35, 28, 20):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            continue
        out = buf.tobytes()
        if len(out) <= max_bytes:
            return out
    return out


def _draw_pole_labels(out: np.ndarray, poles: list[dict],
                      draw_salt_box: bool = True) -> None:
    """Draw L/R status + confidence % at each terminal head."""
    for pole in poles:
        hx, hy, hw, hh = pole["head_rect"]
        pct = pole["confidence"] * 100.0
        is_salt = pole["has_salt"]
        color = (0, 0, 255) if is_salt else (0, 200, 0)
        status = "SALT" if is_salt else "CLEAN"
        tag = f"{pole['name']}: {status} ({pct:.1f}%)"
        ty = max(22, hy - 6)
        cv2.putText(out, tag, (hx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        if is_salt and draw_salt_box:
            if pole.get("box"):
                x, y, bw, bh, _ = pole["box"]
            else:
                x, y, bw, bh = hx, hy, hw, hh
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 3)


def draw_overlay(frame: np.ndarray, result: dict, fps: float = 0.0) -> np.ndarray:
    """Status panel + per-pole % labels; red box only on SALT."""
    out = frame.copy()
    h, w = out.shape[:2]

    has_salt = result["has_salt"]
    conf = result["confidence"]
    color = (0, 0, 255) if has_salt else (0, 200, 0)
    label = "SALT DETECTED" if has_salt else "CLEAN"

    poles = result.get("poles", [])
    if poles:
        _draw_pole_labels(out, poles, draw_salt_box=True)
    elif has_salt:
        for box in result.get("boxes", []):
            x, y, bw, bh, area_ratio = box
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            tag = f"salt {area_ratio*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(0, y - 6)
            cv2.rectangle(out, (x, ty - th - 4), (x + tw + 6, ty), (0, 0, 255), -1)
            cv2.putText(out, tag, (x + 3, ty - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    panel = np.zeros((110, w, 3), dtype=np.uint8)
    cv2.putText(panel, f"{label}  ({conf*100:.1f}%)",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    pole_txt = ""
    if poles:
        pole_txt = "  ".join(
            f"{p['name']}:{'SALT' if p['has_salt'] else 'CLEAN'}"
            f" {p['confidence']*100:.0f}%"
            for p in poles)
        pole_txt += "  "
    cv2.putText(panel,
                pole_txt +
                f"margin: {result['margin']:+.2f}  "
                f"cv: {result['cv_ratio']*100:.1f}%  "
                f"FPS: {fps:.1f}",
                (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)

    return np.vstack([panel, out])


def draw_detecting_banner(frame: np.ndarray, fps: float = 0.0) -> np.ndarray:
    """Banner shown while ONNX inference runs (camera keeps streaming)."""
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 55), (0, 80, 180), -1)
    cv2.putText(out, "DANG QUET MUOI...", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(out, f"FPS: {fps:.1f}", (max(12, w - 130), 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    return out


def draw_live_banner(frame: np.ndarray, fps: float = 0.0,
                     waiting_tb: bool = False,
                     last_result: dict | None = None) -> np.ndarray:
    """Live stream + last scan L/R % if available."""
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = 70 if last_result and last_result.get("poles") else 55
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), -1)
    msg = ("LIVE  |  detect_salt=true de quet"
           if waiting_tb else "LIVE")
    cv2.putText(out, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 220, 0), 2)
    cv2.putText(out, f"FPS: {fps:.1f}", (max(12, w - 130), 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    poles = (last_result or {}).get("poles", [])
    if poles:
        summary = "  ".join(
            f"{p['name']}: {'SALT' if p['has_salt'] else 'CLEAN'}"
            f" ({p['confidence']*100:.1f}%)"
            for p in poles)
        cv2.putText(out, summary, (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        _draw_pole_labels(out, poles, draw_salt_box=False)
    return out


def poles_to_telemetry(result: dict) -> dict:
    """Minimal ThingsBoard telemetry from predict() output."""
    poles = {p["name"]: p for p in result.get("poles", [])}
    pL, pR = poles.get("L", {}), poles.get("R", {})

    def _label(p: dict) -> str:
        return "SALT" if p.get("has_salt") else "CLEAN"

    l_salt = bool(pL.get("has_salt", False))
    r_salt = bool(pR.get("has_salt", False))
    has_salt = (l_salt or r_salt) if poles else bool(result["has_salt"])

    return {
        "terminal_L": _label(pL) if pL else "CLEAN",
        "terminal_R": _label(pR) if pR else "CLEAN",
        "has_salt": "true" if has_salt else "false",
        "trang_thai": "Co muoi" if has_salt else "Khong co muoi",
    }
