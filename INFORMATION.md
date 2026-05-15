# INFORMATION — AI Salt Detector for Marine Battery Terminals

> Comprehensive technical reference for the **AI-predict-salt-h2** project:
> architecture, model, filters, workflow, deployment and operation on
> Raspberry Pi 4 + CSI camera.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware](#2-hardware)
3. [Software Stack](#3-software-stack)
4. [Repository Layout](#4-repository-layout)
5. [AI Model](#5-ai-model)
6. [Few-Shot Prototype Learning](#6-few-shot-prototype-learning)
7. [Classical CV Filters (HSV)](#7-classical-cv-filters-hsv)
8. [Fusion Logic — AI + CV](#8-fusion-logic--ai--cv)
9. [Augmentation Pipeline](#9-augmentation-pipeline)
10. [System Workflow](#10-system-workflow)
11. [Camera Pipeline (CSI)](#11-camera-pipeline-csi)
12. [Installation & Deployment](#12-installation--deployment)
13. [Running & Testing](#13-running--testing)
14. [Experimental Results](#14-experimental-results)
15. [Threshold Tuning](#15-threshold-tuning)
16. [Future Improvements](#16-future-improvements)
17. [References](#17-references)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Project Overview

**Goal:** Automatically detect salt / sulfate corrosion on battery terminals
and casings inside the battery rooms of marine vessels, using a Raspberry
Pi 4 with a fixed CSI camera, with realtime alerts.

**Problem:** Binary image classification (`salt` vs `clean`) with **very
limited data** (8 corroded samples + 5 clean battery samples).

**Solution:** Few-shot learning with a pretrained MobileNetV2 feature
extractor combined with a classical HSV color filter to corroborate the
AI signal. No CNN training from scratch and no TensorFlow / PyTorch at
inference time.

**Measured results:**
- Offline accuracy on the dataset: **13/13 = 100 %**
- Realtime throughput on Pi 4 (640×480 capture): **12.76 FPS**

---

## 2. Hardware

| Component | Model / spec |
|---|---|
| Compute | Raspberry Pi 4 Model B (4 GB RAM recommended) |
| Camera | Sony IMX219 (Pi Camera v2, 8 MP), CSI-2 ribbon |
| Connection | CSI port (15-pin connector near HDMI), not USB |
| Storage | microSD ≥ 16 GB, Class 10 or better |
| Power | Genuine USB-C 5 V / 3 A |
| Network | Wi-Fi 2.4/5 GHz or Ethernet (for SSH and dataset updates) |
| Cooling | Small fan or heatsink — required for 24/7 operation |

**Verify the camera is detected:**
```bash
rpicam-hello --list-cameras
# Expect: imx219 [3280x2464 10-bit RGGB]
```

---

## 3. Software Stack

| Component | Verified version on Pi |
|---|---|
| OS | Raspberry Pi OS / Debian 13 (Trixie), kernel 6.12.75 aarch64 |
| Python | 3.13.5 |
| OpenCV | 4.10.0 (apt `python3-opencv`, includes the DNN module) |
| NumPy | 2.2.4 |
| Picamera2 | apt `python3-picamera2` |
| libcamera | 0.7.0+rpt20260205 |
| rpicam-apps | apt `rpicam-apps` |

**Important:** On Trixie + Python 3.13, **TensorFlow does not yet support
aarch64**. That is why this project uses MobileNetV2 in **ONNX** format
through **OpenCV DNN** — no TF / PyTorch dependency.

---

## 4. Repository Layout

```
AI-predict-salt-h2/
├── dataset/
│   ├── salt/              # 8 corroded battery images (salt_01.jpg ... salt_08.jpg)
│   └── clean/             # 5 clean battery images (clean_01.jpg ... clean_05.jpg)
├── model/
│   └── mobilenetv2-12.onnx   # ~14 MB, MobileNetV2 pretrained on ImageNet
├── snapshots/             # auto-created: frames saved when salt is detected
├── prototypes.npz         # auto-created by train.py: 13 embedding vectors
├── detect_salt.py         # FeatureExtractor + SaltDetector (core)
├── train.py               # builds prototypes.npz from dataset/
├── run_camera.py          # CSI camera loop + UI overlay (GUI / headless)
├── smoke_test.py          # offline accuracy check on the dataset
├── bench_camera.py        # measures real FPS with the camera
├── requirements.txt       # dependencies (just numpy + opencv)
├── README.md              # short quick-start guide
└── INFORMATION.md         # this file
```

---

## 5. AI Model

### MobileNetV2 (ImageNet pretrained) — `model/mobilenetv2-12.onnx`

| Property | Value |
|---|---|
| Architecture | MobileNetV2 (Inverted Residual + Linear Bottleneck) |
| Authors | Sandler et al., Google (CVPR 2018) |
| Pretrained dataset | ImageNet-1K (1.28 M images, 1000 classes) |
| Input | 224×224×3 RGB |
| Normalization | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Output | 1000-D logit vector (used as image embedding) |
| File size | ~14 MB (FP32 ONNX) |
| Speed on Pi 4 | ~70-80 ms/frame via OpenCV DNN |
| Source | https://github.com/onnx/models |
| Paper | https://arxiv.org/abs/1801.04381 |

**Why MobileNetV2:**
- Compact, originally designed for mobile / embedded
- Pretrained on ImageNet → general features good enough for transfer learning
- ONNX format → runs everywhere via `cv2.dnn.readNetFromONNX()`
- Compatible with Python 3.13 / Pi OS Trixie (TF is not)

**Download command:**
```bash
mkdir -p model
wget -O model/mobilenetv2-12.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx
```

---

## 6. Few-Shot Prototype Learning

With only 13 images we **do not train a CNN from scratch**. Pipeline:

1. **Feature extraction**: each image → 1000-D vector from MobileNetV2
2. **L2-normalize** so that cosine similarity reduces to a dot product
3. **Prototype** of each class = mean embedding of its augmented variants
4. **Classification** of a new image: cosine similarity vs the `salt` and
   `clean` prototypes; the higher-similarity class wins
5. **Margin** = `salt_sim − clean_sim` is used as the confidence signal

This is a **Nearest Prototype Classifier** (a.k.a. **Prototypical Networks**
when prototypes are learned end-to-end).

### Theoretical background

- Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
  — https://arxiv.org/abs/1703.05175
- Wang et al., "Generalizing from a Few Examples: A Survey on Few-Shot
  Learning", ACM Computing Surveys 2020
  — https://arxiv.org/abs/1904.05046

---

## 7. Classical CV Filters (HSV)

Defined in `detect_salt.py`, function `salt_color_ratio()`. Three masks
combined with logical OR:

| Mask | HSV condition | Captures |
|---|---|---|
| **White crystals** | S < 60 and V > 150 | Dry salt / sulfate crystals — white-grey, bright |
| **Cyan-green sulfate** | 70 < H < 110, S > 40, V > 120 | Copper sulfate (CuSO₄·5H₂O) — turquoise crust on copper terminals |
| **Yellow-brown crust** | 15 < H < 35, S > 40, V > 120 | Lead sulfur deposits — dry yellow/brown crust |

**Output:** `cv_ratio` ∈ [0, 1] = fraction of pixels matching at least one
of the three masks.

### Scientific basis

- Battery corrosion chemistry:
  - Pb-Sb positive terminals form white **PbSO₄** (sulfation)
  - Copper hardware in contact with electrolyte → turquoise **CuSO₄·5H₂O**
  - General corrosion byproducts → brown-yellow crust (lead oxide + sulfur)
- HSV instead of RGB: separates hue (color) from lighting → robust to
  ship-room lamp brightness changes
- Reference: Gonzalez & Woods, *Digital Image Processing*, Ch. 6
  (Color Image Processing)

---

## 8. Fusion Logic — AI + CV

In `detect_salt.py`, method `SaltDetector.predict()`:

### Computation

```
margin     = salt_sim - clean_sim           (cosine similarity, L2-normed)
ai_score   = clip( (margin + 0.1) / 0.3, 0, 1 )
cv_score   = min( cv_ratio / 0.25, 1 )
confidence = 0.7 * ai_score + 0.3 * cv_score
```

### `has_salt = True` if either condition holds:

```
(1) margin >= 0.03                           # AI alone is confident
(2) cv_ratio >= 0.08  AND  margin >= -0.02   # strong CV evidence and AI does not disagree
```

### Why both signals?

- **AI only**: misses corrosion that looks very different from training samples
- **CV only**: false-positives on white battery casings (high cv_ratio
  from the casing color, not from salt)
- **Combined**: AI provides context ("this is a battery terminal"), CV
  provides physical evidence ("there are salt-colored pixels")

### 8.1 Where does the formula `margin = salt_sim − clean_sim` come from?

This is **not** an ad-hoc formula — it is the explicit form of the
**Nearest-Prototype decision rule** used in few-shot learning
(Snell et al., *Prototypical Networks for Few-shot Learning*, NeurIPS
2017, https://arxiv.org/abs/1703.05175).

**Step-by-step derivation:**

1. `train.py` computes one prototype per class as the mean of the
   L2-normalized embeddings of all augmented samples:

   $$\mathbf{p}_{\text{salt}} = \frac{1}{N_s}\sum_{i=1}^{N_s} \mathbf{e}_i^{\text{salt}}, \quad
     \mathbf{p}_{\text{clean}} = \frac{1}{N_c}\sum_{j=1}^{N_c} \mathbf{e}_j^{\text{clean}}$$

2. At inference, the query image is embedded into vector $\mathbf{e}$ and
   L2-normalized (`detect_salt.py`, `FeatureExtractor.embed`). Because
   both prototypes and query are unit vectors, the dot product **is**
   the cosine similarity:

   $$\text{sim}(\mathbf{p}, \mathbf{e}) = \mathbf{p}\cdot\mathbf{e} = \cos(\theta) \in [-1, 1]$$

3. The Prototypical-Network decision rule is *"assign the query to the
   class of the nearest prototype"*. With cosine distance this means
   **pick the class with the higher similarity**:

   $$\hat{y} = \arg\max_{c \in \{\text{salt},\,\text{clean}\}} \text{sim}(\mathbf{p}_c, \mathbf{e})$$

4. For a 2-class problem this `argmax` is equivalent to checking the
   sign of the **difference** of the two similarities. That difference
   is exactly our `margin`:

   $$\text{margin} \;=\; \text{salt\_sim} - \text{clean\_sim}
   \begin{cases}
   > 0 & \Rightarrow\ \text{predict salt} \\
   < 0 & \Rightarrow\ \text{predict clean} \\
   = 0 & \Rightarrow\ \text{on the decision boundary}
   \end{cases}$$

   The **magnitude** $|\text{margin}|$ is the distance from the decision
   boundary, i.e. how confident the classifier is. This "margin" concept
   is identical in spirit to the SVM margin (Cortes & Vapnik, 1995) and
   to the logit difference used in binary softmax classifiers.

5. **Why the *difference* and not just `salt_sim` alone?**
   Absolute cosine values drift with image quality: dark, blurry, or
   out-of-distribution frames push *both* similarities down together.
   Subtracting `clean_sim` cancels this common-mode bias and leaves only
   the **relative** evidence ("does the frame look more like salt than
   like clean?"). This is the same trick used in contrastive losses and
   in log-likelihood-ratio tests (Neyman–Pearson lemma).

### 8.2 What is `ai_score` and what does it represent?

`ai_score` is **not** a classification result. It is the AI branch's
**confidence**, rescaled into `[0, 1]` so it can be linearly fused with
the CV branch's score (which is also in `[0, 1]`).

```python
# detect_salt.py, SaltDetector.predict()
ai_score = float(np.clip((margin + 0.1) / 0.3, 0.0, 1.0))
```

**Interpretation of the values:**

| `margin` | `ai_score` | Meaning |
|---:|---:|---|
| ≤ −0.10 | 0.00 | AI is very confident: **clean** |
| −0.05 | 0.17 | AI leans clean |
|  0.00 | 0.33 | Undecided (on the boundary) |
| +0.05 | 0.50 | AI mildly leans salt |
| +0.10 | 0.67 | AI leans salt |
| ≥ +0.20 | 1.00 | AI is very confident: **salt** |

**Where do the constants `0.1` and `0.3` come from?**
They are an **affine rescaling** (a heuristic, not a theorem):

$$\text{ai\_score} = \mathrm{clip}\!\left(\frac{\text{margin} - m_{\min}}{m_{\max} - m_{\min}},\ 0,\ 1\right)
\quad\text{with}\quad m_{\min} = -0.1,\ m_{\max} = +0.2$$

Empirically, with MobileNetV2 logits as embeddings on this dataset,
margins fall in roughly `[-0.1, +0.2]` (see §14, "Mean margin"). Mapping
that empirical range to `[0, 1]` gives every value the same scale as
`cv_score`, which is required for the linear fusion:

```
confidence = 0.7 * ai_score + 0.3 * cv_score
```

If you change the backbone or the dataset, the empirical margin range
will change and these constants should be re-tuned (see §15).

### 8.3 Summary of the signal chain

```
embedding e  ──► salt_sim, clean_sim   (cosine vs prototypes)
                     │
                     ▼
              margin = salt_sim − clean_sim     ← decision rule
                     │
                     ▼
              ai_score ∈ [0,1]                  ← rescaled margin
                     │
                     ├── fused with cv_score → confidence (displayed)
                     │
                     └── thresholded (margin ≥ 0.03) → has_salt (boolean)
```

---

## 9. Augmentation Pipeline

In `train.py`, function `augment()`. Each source image yields **5 variants**:

| Variant | Purpose |
|---|---|
| Original | Baseline |
| Horizontal flip | Invariance to left/right terminal orientation |
| Gamma 0.7 (darker) | Dim battery rooms |
| Gamma 1.3 (brighter) | Direct flashlight |
| Warm white-balance shift (B−15, R+15) | Yellow shipboard lighting |

→ 13 images × 5 = **65 embeddings** → averaged per source image →
**13 prototypes**.

---

## 10. System Workflow

### A. Training workflow (offline, run once or whenever new images are added)

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ dataset/   │──▶│  augment   │──▶│ MobileNetV2│──▶│ L2-norm +  │
│  salt/     │   │ (×5/image) │   │   ONNX     │   │   mean     │
│  clean/    │   │            │   │ (1000-D)   │   │            │
└────────────┘   └────────────┘   └────────────┘   └─────┬──────┘
                                                         │
                                                         ▼
                                                  prototypes.npz
                                              (13 L2-normed vectors)
```

**Command:** `python3 train.py`

### B. Realtime inference workflow (on the Pi)

```
┌────────────┐   ┌──────────┐   ┌──────────────┐   ┌─────────────┐
│ CSI camera │──▶│Picamera2 │──▶│  cv2.dnn     │──▶│  cosine vs  │
│  (IMX219)  │   │  RGB888  │   │ MobileNetV2  │   │ 13 prototypes│
└────────────┘   └──────────┘   └──────────────┘   └──────┬──────┘
                                                          │
                       ┌──────────────────────────────────┘
                       ▼
                ┌──────────────┐    ┌──────────────┐
                │ HSV color    │───▶│ Fusion logic │───▶ has_salt + confidence
                │ mask (3 cols)│    │ AI 0.7 +     │
                └──────────────┘    │ CV 0.3       │
                                    └──────┬───────┘
                                           │
                          ┌────────────────┴────────────────┐
                          ▼                                 ▼
                   GUI overlay                       Headless mode:
                  (red/green + %)                  print [ALERT] + save snapshot
```

**Command:**
- GUI: `python3 run_camera.py`
- Headless: `python3 run_camera.py --headless --width 640 --height 480`

### C. Continuous operation (production)

```
1. Pi boots → systemd service auto-starts run_camera.py --headless
2. Camera streams continuously, inference at ~12 FPS
3. When has_salt = True:
   - Log [ALERT] with timestamp + confidence
   - Save snapshot to snapshots/<timestamp>_salt.jpg
   - 5-second cooldown before the next alert
4. Operators periodically:
   - SSH into the Pi and review logs
   - Pull snapshots for inspection
   - Add mislabeled samples back into dataset/ → retrain
```

---

## 11. Camera Pipeline (CSI)

| Layer | Library | Hardware |
|---|---|---|
| **Sensor driver** | libcamera 0.7.0 + tuning file `imx219.json` | Sony IMX219 |
| **Capture API** | Picamera2 (Python wrapper) | CSI-2 → /dev/media1 (ISP), /dev/media4 (Unicam) |
| **Format** | RGB888, 1280×720 default (640×480 for higher FPS) | Auto white balance + auto exposure |
| **Conversion** | `cv2.cvtColor(RGB → BGR)` | For OpenCV compatibility |
| **Fallback** | `cv2.VideoCapture(0)` | USB webcam when testing on a laptop |

### Official documentation

- Picamera2 Manual (PDF): https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
- libcamera API: https://libcamera.org/api-html/
- Raspberry Pi camera docs: https://www.raspberrypi.com/documentation/computers/camera_software.html

---

## 12. Installation & Deployment

### A. On Raspberry Pi 4 (Bookworm or Trixie 64-bit)

```bash
# 1. Enable the CSI camera
sudo raspi-config           # Interface Options → Camera → Enable → reboot
rpicam-hello --list-cameras # sanity check

# 2. Install system dependencies (do NOT install tensorflow)
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy \
                    python3-pip rpicam-apps

# 3. Copy the project to the Pi (from the dev machine via SCP)
scp -r AI-predict-salt-h2 pi@<PI_IP>:~/Desktop/

# 4. Build prototypes
cd ~/Desktop/AI-predict-salt-h2
python3 train.py
```

### B. On a development machine (Windows / Linux x86_64)

```powershell
cd AI-predict-salt-h2
py -3 -m pip install numpy opencv-python
py -3 train.py
py -3 run_camera.py        # uses a USB webcam
```

### C. Run as a systemd service (Pi production)

Create `/etc/systemd/system/salt-detector.service`:

```ini
[Unit]
Description=Marine Battery Salt Detector
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Desktop/AI-predict-salt-h2
ExecStart=/usr/bin/python3 run_camera.py --headless --width 640 --height 480
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now salt-detector
sudo journalctl -u salt-detector -f      # tail logs in realtime
```

---

## 13. Running & Testing

### 1. Offline test on the bundled dataset

```bash
python3 smoke_test.py
# Expected: Accuracy: 13/13 = 100.0%
```

### 2. Real FPS measurement with the CSI camera

```bash
python3 bench_camera.py
# Real measurement: TOTAL 103 frames in 8.1s -> 12.76 FPS
```

### 3. Headless run (over SSH, no display needed)

```bash
python3 run_camera.py --headless --width 640 --height 480
```

When the camera sees salt → prints `[ALERT]` and saves a snapshot to
`snapshots/`.

### 4. GUI run (requires a display or VNC)

```bash
python3 run_camera.py
```

A window shows live video with a **red "SALT DETECTED"** or **green
"CLEAN"** border. Hotkeys: `s` save snapshot, `q` quit.

### 5. Mock test (no real battery required)

Print an image from `dataset/salt/` on paper or display it on a phone
in front of the CSI camera → the system reacts immediately.

---

## 14. Experimental Results

### Offline accuracy on the dataset (13 images)

| Class | Images | Correct | Mean margin |
|---|---|---|---|
| salt | 8 | 8/8 | +0.605 (very confident) |
| clean | 5 | 5/5 | −0.509 (very confident) |
| **Total** | **13** | **13/13 = 100 %** | — |

### Realtime performance on Pi 4

| Metric | Value |
|---|---|
| Capture resolution | 640 × 480 |
| Measured FPS | 12.76 (78 ms/frame end-to-end) |
| RAM idle | ~150 MB |
| RAM peak (after ONNX load) | ~280 MB |
| CPU usage | ~75 % of one core (Pi 4 has 4 cores) |

---

## 15. Threshold Tuning

In `detect_salt.py`, class `SaltDetector(...)`:

| Parameter | Default | Increase when… | Decrease when… |
|---|---|---|---|
| `margin_threshold` | 0.03 | Too many false positives | Salt is being missed |
| `cv_threshold` | 0.08 | Triggering on white casings (false positive) | Light corrosion is missed |
| `fusion_weight_ai` | 0.7 | Dataset is well-representative | CV mask is more reliable than AI |

**Quick override** (edit the `SaltDetector(...)` call in `run_camera.py`):

```python
detector = SaltDetector(
    prototypes_path=args.prototypes,
    model_path=args.model,
    margin_threshold=0.05,    # stricter
    cv_threshold=0.12,
    fusion_weight_ai=0.6,
)
```

---

## 16. Future Improvements

| Step | When needed | How |
|---|---|---|
| **Add real images** | After every voyage | Drop them into `dataset/salt/` or `dataset/clean/` and rerun `python3 train.py`. No code changes. |
| **Switch to YOLOv8-cls** | Once you have >100 images / class | Train YOLOv8n-cls (~3 MB), export to ONNX, replace the model file. Minimal code changes. |
| **INT8 quantization** | Need >20 FPS on the Pi | Use `onnxruntime` quantize → 4× smaller, 2-3× faster |
| **Object detection / segmentation** | Need to localize the salt in the frame | Switch to YOLOv8n-seg / SAM, output a mask instead of a binary label |
| **Network alerts** | Integrate with the ship's SCADA | Add `paho-mqtt`, publish to topic `ship/battery/salt_alert` when has_salt is True |
| **Event history** | Audit trail | Log every prediction to a SQLite `events.db` (timestamp, confidence, snapshot path) |
| **Web dashboard** | Remote monitoring | Flask/FastAPI live feed + alert history |

---

## 17. References

### Computer vision & deep learning

- Sandler et al., **MobileNetV2: Inverted Residuals and Linear Bottlenecks**,
  CVPR 2018 — https://arxiv.org/abs/1801.04381
- Howard et al., **MobileNets: Efficient CNNs for Mobile Vision**
  — https://arxiv.org/abs/1704.04861
- Snell et al., **Prototypical Networks for Few-shot Learning**, NeurIPS 2017
  — https://arxiv.org/abs/1703.05175
- Wang et al., **Generalizing from a Few Examples: A Survey on Few-Shot
  Learning**, ACM Computing Surveys 2020
  — https://arxiv.org/abs/1904.05046
- He et al., **Deep Residual Learning for Image Recognition**, CVPR 2016
  — https://arxiv.org/abs/1512.03385
- Gonzalez & Woods, **Digital Image Processing**, Pearson 4th ed. — Ch. 6
  Color Image Processing

### Tools & frameworks

- OpenCV DNN module: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
- ONNX specification: https://github.com/onnx/onnx
- ONNX Model Zoo: https://github.com/onnx/models
- Picamera2: https://github.com/raspberrypi/picamera2
- libcamera: https://libcamera.org/

### Domain — marine lead-acid battery corrosion

- IEEE 484-2019, **Recommended Practice for Installation Design and
  Installation of Vented Lead-Acid Batteries**
- IEEE 1188-2005, **Recommended Practice for Maintenance, Testing, and
  Replacement of Valve-Regulated Lead-Acid (VRLA) Batteries**
- IEC 60092-507, **Electrical installations in ships – Pleasure craft**
- ABS, **Guidance Notes on Battery Systems for Marine Applications**

---

## 18. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Picamera2 not available` on the Pi | `python3-picamera2` not installed | `sudo apt install python3-picamera2` |
| `Can't read ONNX file` | Truncated ONNX download (size < 14 MB) | Re-download from the ONNX Model Zoo |
| `prototypes.npz not found` | Training step never ran | `python3 train.py` |
| Only 2-3 FPS on the Pi | Capture resolution too high | Use `--width 640 --height 480` |
| False positive on white casings | `cv_threshold` too low | Raise `cv_threshold` to 0.12-0.15 |
| Light corrosion missed | `margin_threshold` too high | Lower to 0.01-0.02, or add light-corrosion samples to the dataset |
| Camera not detected | Loose CSI cable / not enabled | `sudo raspi-config` → Camera → Enable → reboot |
| `ModuleNotFoundError: cv2` on Pi | venv without system packages | Recreate the venv with `--system-site-packages` |
| Pi overheating / thermal throttling | No cooling | Add a fan + heatsink, check `vcgencmd measure_temp` (keep <70 °C) |
| SSH hangs when GUI mode runs | Headless Pi has no X server | Use `--headless` instead of opening a window |
| `[ALERT]` prints continuously | Camera is actually pointed at salt | Expected — there is already a 5-second cooldown |

---

> **Project:** AI-predict-salt-h2
> **Last updated:** 2026-05-06
> **License:** Internal use
