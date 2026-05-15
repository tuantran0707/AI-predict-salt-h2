# Salt / sulfate corrosion detector for marine batteries (Raspberry Pi 4 + CSI camera)

Detects salt and sulfate corrosion on battery terminals and casings on
board ships using a Raspberry Pi 4 with a CSI camera. Because the
training set is tiny (a handful of close-ups of corroded terminals plus a
handful of clean battery shots) the detector uses **few-shot learning**
instead of training a CNN from scratch:

1. `train.py` runs every image in `dataset/salt/` and `dataset/clean/`
   through MobileNetV2 (pretrained on ImageNet) with light augmentations
   and stores one mean L2-normalized embedding per image into
   `prototypes.npz`.
2. `run_camera.py` reads frames from the CSI camera, embeds each frame
   and compares it against both prototype banks via cosine similarity.
   The class with the higher similarity wins; an HSV color mask
   (white crystals / cyan-green copper sulfate / yellow crust) is fused
   in as an independent corroboration signal.

## Project layout

```
AI-predict-salt-h2/
├── dataset/
│   ├── salt/      # battery terminals with salt / sulfate corrosion
│   └── clean/     # clean battery terminals (negative samples)
├── detect_salt.py # FeatureExtractor + SaltDetector
├── train.py       # builds prototypes.npz from dataset/
├── run_camera.py  # CSI camera loop + UI overlay (with headless mode)
├── requirements.txt
└── README.md
```

## Install on a development machine (Windows / Linux x86_64)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py            # builds prototypes.npz
python run_camera.py       # tests with the default USB webcam
```

## Install on Raspberry Pi 4 (Raspberry Pi OS Bookworm 64-bit)

Enable the CSI camera first:

```bash
sudo raspi-config           # Interface Options -> Camera -> Enable, then reboot
libcamera-hello -t 2000     # sanity check the camera
```

Install system packages and create a venv that can see them:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-pip python3-venv

cd ~/AI-predict-salt-h2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy
pip install tensorflow-aarch64    # full TF for aarch64
# (or, if RAM is tight: pip install tflite-runtime — would require
#  exporting MobileNetV2 to TFLite and adapting FeatureExtractor.)

python train.py
python run_camera.py                    # GUI mode (needs an X server)
# Headless variants:
python run_camera.py --headless         # CSI camera, no GUI
python run_camera.py --headless --no-csi --width 640 --height 480   # USB cam
```

### Performance tips for Pi 4

- Inference runs every 5 frames by default (`--infer-every 5`) so the
  preview stays smooth. Increase to 10 if CPU is saturated.
- Lower the capture resolution if needed: `--width 640 --height 480`.
- For best throughput, switch to a TFLite model (MobileNetV2 quantized
  to int8 runs at ~30–50 ms/frame on a Pi 4 with `tflite-runtime`).
- Make sure the Pi has good cooling — sustained TF inference will
  thermal-throttle a fanless Pi 4.

## Tuning thresholds

Edit `SaltDetector(...)` in `run_camera.py` or in your own script:

| Parameter | Default | Meaning |
|---|---|---|
| `margin_threshold` | 0.025 | Minimum (salt_sim − clean_sim) to flag salt from AI alone |
| `cv_threshold` | 0.03 | Minimum HSV salt-mask ratio that on its own raises suspicion |
| `fusion_weight_ai` | 0.7 | Weight of the AI score when fusing with the CV score |

When you collect more real images on the ship, drop them into the
appropriate `dataset/salt/` or `dataset/clean/` folder and rerun
`python train.py`. No code changes required.

## Hotkeys (GUI mode)

- `q` — quit
- `s` — save a snapshot to `snapshots/`

## Headless mode

`python run_camera.py --headless` runs without any window. Frames where
salt is detected are auto-saved to `snapshots/` (rate-limited to one
every 5 seconds) and printed to stdout — handy when running the Pi as a
background monitor with `systemd` or `tmux`.
