from pathlib import Path
import numpy as np
from .config_labels import LABEL_MAP

def collect_img_paths_with_labels(data_root):
    img_paths_label = []
    classes = set()

    for cls in LABEL_MAP.keys():
        cls_dir = Path(data_root) / cls
        for img_path in cls_dir.glob("*.*"):
            img_paths_label.append((img_path, LABEL_MAP[cls]["family"]))

    print(f"Collected {len(img_paths_label)} img_paths_label from {len(LABEL_MAP)} classes.")

    return img_paths_label

def extract_patches(img, patch_size=128, stride=64):
    patches = []
    h, w = img.shape[:2]

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)

    return patches