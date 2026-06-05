# INFORMATION - AI Salt Detector for Marine Battery Terminals

Technical reference for the AI-predict-salt-h2 project: model architecture,
training workflow, tuned coefficients, and deployment notes.

## 1. Project Summary

Goal: detect battery terminal corrosion (salt/sulfate) vs clean terminals from
camera images in marine environments.

Approach:
- Few-shot classification using MobileNetV2 ONNX embeddings.
- Nearest-prototype decision with cosine similarity.
- HSV-based CV signal used as corroboration, not primary decision.

Current dataset status:
- Raw files in dataset: 338 (salt: 109, clean: 229)
- Unreadable files: 10
- Usable images for training/evaluation: 328 (salt: 105, clean: 223)

## 2. Repository Layout

```text
AI-predict-salt-h2/
|- dataset/
|  |- salt/
|  `- clean/
|- model/
|  `- mobilenetv2-12.onnx
|- detect_salt.py
|- train.py
|- run_camera.py
|- smoke_test.py
|- bench_camera.py
|- prototypes.npz
|- README.md
`- INFORMATION.md
```

## 3. Model and Feature Extraction

Backbone:
- MobileNetV2 pretrained on ImageNet, loaded from ONNX through OpenCV DNN.

Input preprocessing:
- Resize to 224x224 RGB
- Normalize with ImageNet mean/std
- Output embedding: 1000-D logits, then L2 normalization

Reasoning:
- MobileNetV2 is lightweight and reliable on edge hardware.
- ONNX + OpenCV DNN avoids TensorFlow dependency issues on Raspberry Pi setups.

## 4. Training (Prototype Building)

Script: train.py

Pipeline:
1. Read images from dataset/salt and dataset/clean.
2. Generate 5 variants per valid image:
   - original
   - horizontal flip
   - gamma 0.7
   - gamma 1.3
   - warm white-balance shift
3. Embed all variants.
4. Average embeddings per source image.
5. L2-normalize and store in prototypes.npz.

Current result after retraining:
- Saved prototypes: 328 total (salt: 105, clean: 223)
- Unreadable images are skipped automatically.

## 5. Inference Logic (detect_salt.py)

### 5.1 Similarity and margin

For each input image:
- salt_sim = max cosine similarity to salt prototypes
- clean_sim = max cosine similarity to clean prototypes
- margin = salt_sim - clean_sim

Primary decision is margin-based.

### 5.2 Tuned coefficients (current)

In SaltDetector defaults:
- margin_threshold = 0.04
- cv_threshold = 0.15
- cv_margin_guard = 0.02
- fusion_weight_ai = 0.9

Score mapping:
- ai_score = clip((margin + 0.40) / 0.92, 0, 1)
- cv_score = min(cv_ratio / 0.58, 1)
- confidence = 0.9 * ai_score + 0.1 * cv_score

Boolean decision:
- has_salt = (margin >= 0.04)
  OR (cv_ratio >= 0.15 AND margin >= 0.02)

Why this tuning:
- Margin separates classes very well on current data.
- CV color ratio alone is noisy on clean battery surfaces, so CV weight is reduced.
- CV is kept only as corroboration for borderline positives.

## 6. Empirical Statistics Used for Tuning

Computed on current usable dataset (328 images):

Margin distributions:
- salt: mean 0.3353, std 0.1049, P5 0.2125, P95 0.5184
- clean: mean -0.2654, std 0.0748, P5 -0.4021, P95 -0.1589

CV ratio distributions:
- salt: mean 0.3044, P95 0.5805
- clean: mean 0.4578, P95 0.8591

Interpretation:
- Margin is the reliable separator.
- CV ratio has overlap/noise, therefore lower influence in fusion.

## 7. Validation Result (after retune)

Validation on all readable images:
- valid_images = 328
- accuracy = 328/328 (100.00%)
- misclassified = 0
- unreadable = 10

Note:
- This is in-dataset validation, useful for sanity check.
- Real deployment performance depends on camera angle, lighting, and domain shift.

## 8. Unreadable Files

These files are present but cannot be loaded by OpenCV:

- dataset/salt/salt_0011.jpg
- dataset/salt/salt_0049.jpg
- dataset/salt/salt_0054.jpg
- dataset/salt/salt_0060.jpg
- dataset/clean/clean_0086.jpg
- dataset/clean/clean_0088.jpg
- dataset/clean/clean_0178.png
- dataset/clean/clean_0182.png
- dataset/clean/clean_0196.png
- dataset/clean/clean_0197.png

Recommendation:
- Replace or remove these files to keep dataset quality stable.

## 9. Runtime Notes (run_camera.py)

Default behavior:
- Load prototypes.npz and MobileNetV2 ONNX
- Run inference every N frames (default infer-every = 5)
- Display:
  - SALT DETECTED / CLEAN
  - confidence
  - salt_sim, clean_sim, margin, cv_ratio
  - optional bounding boxes from HSV mask

Headless mode:
- Saves alert snapshots with cooldown when salt is detected.

## 10. Recommended Operating Procedure

1. Add new field images into correct class folder.
2. Retrain:
   - Windows: py -3.13 train.py
   - Linux/Pi: python3 train.py
3. Validate quickly on dataset.
4. Deploy and monitor false positives/false negatives.
5. Re-tune thresholds only when deployment data changes noticeably.

## 11. Quick Troubleshooting

If training fails with syntax error at type hints:
- You are likely running Python 2 by mistake.
- Use explicit interpreter:
  - py -3.13 train.py (Windows)
  - python3 train.py (Linux/Pi)

If model loads but detection is unstable:
- Reduce lighting variance.
- Ensure terminal area occupies enough pixels.
- Add representative real-world images and retrain.
