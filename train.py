"""
train.py
--------
Few-shot "training" for the salt detector.

Reads two folders:
    dataset/salt/    -> battery terminals with salt / sulfate corrosion
    dataset/clean/   -> clean battery terminals (negative class)

For each image we generate a few light augmentations (flip, gamma, white
balance shift) and average their MobileNetV2 embeddings into a single
L2-normalized prototype vector. All prototypes are saved to prototypes.npz.

When you collect more real images on the ship, just drop them into the
right subfolder of dataset/ and rerun this script.
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from detect_salt import FeatureExtractor, save_prototypes

DATASET_DIR = "dataset"
OUT_PATH = "prototypes.npz"
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def list_images(folder: str) -> list[str]:
    paths: list[str] = []
    for ext in EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def augment(bgr: np.ndarray) -> list[np.ndarray]:
    """Generate a few light variants of the input image."""
    out = [bgr, cv2.flip(bgr, 1)]

    # Brightness / gamma
    for gamma in (0.7, 1.3):
        lut = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        ).astype("uint8")
        out.append(cv2.LUT(bgr, lut))

    # Mild warm white-balance shift (yellow shipboard lighting)
    warm = bgr.copy().astype(np.int16)
    warm[..., 0] = np.clip(warm[..., 0] - 15, 0, 255)   # less blue
    warm[..., 2] = np.clip(warm[..., 2] + 15, 0, 255)   # more red
    out.append(warm.astype(np.uint8))

    return out


def build_prototypes(folder: str, extractor: FeatureExtractor, label: str):
    paths = list_images(folder)
    if not paths:
        print(f"  [!] No images in '{folder}'.")
        return np.zeros((0, 1280), dtype=np.float32), []

    embeddings: list[np.ndarray] = []
    names: list[str] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [!] Skipped {p} (could not read).")
            continue
        variants = augment(img)
        vecs = np.stack([extractor.embed(v) for v in variants])
        proto = vecs.mean(axis=0)
        proto /= (np.linalg.norm(proto) + 1e-9)
        embeddings.append(proto)
        names.append(os.path.basename(p))
        print(f"  [+] [{label}] {os.path.basename(p)}  ({len(variants)} variants)")

    return np.stack(embeddings).astype(np.float32), names


def main() -> None:
    salt_dir = os.path.join(DATASET_DIR, "salt")
    clean_dir = os.path.join(DATASET_DIR, "clean")
    if not (os.path.isdir(salt_dir) and os.path.isdir(clean_dir)):
        raise SystemExit(
            f"Expected dataset layout:\n  {salt_dir}/\n  {clean_dir}/"
        )

    print("[i] Loading MobileNetV2 (first run downloads ~14 MB of weights)...")
    extractor = FeatureExtractor()

    print("[i] Building prototypes for class 'salt'...")
    salt_protos, salt_names = build_prototypes(salt_dir, extractor, "salt")

    print("[i] Building prototypes for class 'clean'...")
    clean_protos, clean_names = build_prototypes(clean_dir, extractor, "clean")

    if len(salt_protos) == 0 or len(clean_protos) == 0:
        raise SystemExit("Both classes must contain at least one image.")

    save_prototypes(OUT_PATH, salt_protos, clean_protos, salt_names, clean_names)
    print(f"[OK] Saved {len(salt_protos)} salt + {len(clean_protos)} clean "
          f"prototypes to {OUT_PATH}")


if __name__ == "__main__":
    main()
