# AI nhận diện muối bám trên thùng acquy (Raspberry Pi 4 + CSI camera)

Phát hiện muối / cặn sulfat bám trên cực và thùng acquy của tàu biển bằng
camera CSI gắn trực tiếp vào Pi 4. Vì hiện chỉ có **5 ảnh mẫu** nên dùng
**few-shot learning**:

1. `train.py` dùng MobileNetV2 (pretrained ImageNet) trích đặc trưng cho
   5 ảnh trong `images/` (kèm augment nhẹ) và lưu thành `prototypes.npz`.
2. `run_camera.py` đọc frame từ CSI camera, tính embedding của frame, so
   sánh cosine similarity với 5 prototype + kiểm tra mask màu (HSV) đặc
   trưng của muối/sulfat → cho ra cảnh báo `CÓ MUỐI` / `SẠCH`.

## Cấu trúc

```
AI-predict-salt-h2/
├── images/              # 5 ảnh mẫu salt_01..salt_05.jpg
├── train.py             # build prototypes.npz
├── detect_salt.py       # FeatureExtractor + SaltDetector
├── run_camera.py        # vòng lặp camera CSI + UI
├── requirements.txt
└── README.md
```

## Cài đặt trên máy phát triển (Windows / Linux)

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
python train.py            # tạo prototypes.npz
python run_camera.py       # test bằng webcam USB
```

## Cài đặt trên Raspberry Pi 4 (Bookworm 64-bit)

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-pip python3-venv

# Tạo venv kế thừa picamera2 từ hệ thống
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# TensorFlow cho aarch64
pip install numpy
pip install tensorflow-aarch64        # hoặc: pip install tflite-runtime

# Copy project sang Pi rồi:
python train.py
python run_camera.py
```

> Bật camera CSI: `sudo raspi-config` → Interface Options → Camera → Enable
> rồi reboot. Kiểm tra: `libcamera-hello -t 2000`.

## Tinh chỉnh ngưỡng

Trong `detect_salt.py`, lớp `SaltDetector`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `ai_threshold` | 0.55 | Cosine similarity tối thiểu để coi là khớp mẫu |
| `cv_threshold` | 0.08 | Tỉ lệ pixel "muối" tối thiểu trong khung hình |
| `fusion_weight_ai` | 0.7 | Trọng số AI khi hợp nhất với CV |

Khi thu thêm ảnh thật trên tàu, bỏ vào `images/` rồi chạy lại `python train.py`.
Càng nhiều mẫu, prototype càng đại diện và độ chính xác càng tăng.

## Phím trong cửa sổ camera
- `q` — thoát
- `s` — lưu snapshot vào `snapshots/`
