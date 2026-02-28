import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from config_labels import LABEL_MAP
from datasets import MicrostructurePatchDataset
from pathlib import Path


def custom_collate(batch):
    rgb = torch.stack([b["rgb"] for b in batch], dim=0)

    labels = torch.stack([b["label"] for b in batch], dim=0)
    coords = torch.stack([b["coords"] for b in batch], dim=0)

    img_paths = [b["img_path"] for b in batch]  # keep as list of strings

    lbp_list = [b["lbp"] for b in batch]

    # If ANY sample has None → whole batch LBP = None
    if any(l is None for l in lbp_list):
        lbp = None
    else:
        lbp = torch.stack(lbp_list, dim=0)

    return {
        "rgb": rgb,
        "lbp": lbp,
        "label": labels,
        "img_path": img_paths,
        "coords": coords,
    }


def collect_img_paths_with_labels(data_root):
    img_paths_label = []
    classes = set()

    for cls in LABEL_MAP.keys():
        cls_dir = Path(data_root) / cls
        for img_path in cls_dir.glob("*.*"):
            img_paths_label.append((img_path, LABEL_MAP[cls]["family"]))

    print(
        f"Collected {len(img_paths_label)} img_paths_label from {len(LABEL_MAP)} classes."
    )

    return img_paths_label


def make_train_val_test_loaders(
    data_root,
    batch_size=128,
    patch_size=128,
    stride=64,
    lbp_settings=[(1, 8), (3, 16)],
    val_frac=0.15,
    test_frac=0.15,
    num_workers=0,  # Windows Jupyter safe - multiprocessing issue
    random_state=42,
):
    img_paths_label = collect_img_paths_with_labels(data_root)

    print(f"Collected {len(img_paths_label)} images")

    # Stratify by family (0=UB, 1=Lath-type)
    y_family = np.array([s[1] for s in img_paths_label])

    # -----------------------
    # Split off TEST set
    # -----------------------
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=random_state
    )

    trainval_idx, test_idx = next(sss_test.split(img_paths_label, y_family))

    trainval_samples = [img_paths_label[i] for i in trainval_idx]
    test_samples = [img_paths_label[i] for i in test_idx]

    y_trainval = y_family[trainval_idx]

    # -----------------------
    # Split TRAIN / VAL
    # -----------------------
    val_size_rel = val_frac / (1.0 - test_frac)

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size_rel, random_state=random_state
    )

    train_idx, val_idx = next(sss_val.split(trainval_samples, y_trainval))

    train_samples = [trainval_samples[i] for i in train_idx]
    val_samples = [trainval_samples[i] for i in val_idx]

    # -----------------------
    # Datasets
    # -----------------------
    train_ds = MicrostructurePatchDataset(
        train_samples,
        patch_size=patch_size,
        stride=stride,
        augment=True,
        lbp_settings=lbp_settings,
    )

    val_ds = MicrostructurePatchDataset(
        val_samples,
        patch_size=patch_size,
        stride=stride,
        augment=False,
        lbp_settings=lbp_settings,
    )

    test_ds = MicrostructurePatchDataset(
        test_samples,
        patch_size=patch_size,
        stride=stride,
        augment=False,
        lbp_settings=lbp_settings,
    )

    # -----------------------
    # Loaders
    # -----------------------
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers == 0),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=custom_collate,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers == 0),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=custom_collate,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers == 0),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=custom_collate,
    )

    print("Train family distribution:", np.bincount(y_family[train_idx]))
    print("Val family distribution:", np.bincount(y_family[val_idx]))
    print("Test family distribution:", np.bincount(y_family[test_idx]))
    return train_loader, val_loader, test_loader


def make_loader_from_samples(
    samples, patch_size, stride, augment, lbp_settings, batch_size=128, num_workers=1, shuffle=True
):
    ds = MicrostructurePatchDataset(
        samples,
        patch_size=patch_size,
        stride=stride,
        augment=True,
        lbp_settings=lbp_settings,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(num_workers == 0),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=custom_collate,
    )
    return loader


def make_cv_loaders_from_samples(
    samples,
    batch_size=128,
    n_splits=4,
    patch_size=256,
    stride=64,
    lbp_settings=[(1, 8), (3, 16)],
    num_workers=0,
):
    y_family = np.array([s[1] for s in samples])

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(samples, y_family)):

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_loader = make_loader_from_samples(
            train_samples, patch_size, stride, True, lbp_settings
        )
        val_loader = make_loader_from_samples(
            val_samples, patch_size, stride, True, lbp_settings
        )
        # train_ds = MicrostructurePatchDataset(
        #     train_samples,
        #     patch_size=patch_size,
        #     stride=stride,
        #     augment=True,
        #     lbp_settings=lbp_settings,
        # )

        # val_ds = MicrostructurePatchDataset(
        #     val_samples,
        #     patch_size=patch_size,
        #     stride=stride,
        #     augment=False,
        #     lbp_settings=lbp_settings,
        # )

        # train_loader = torch.utils.data.DataLoader(
        #     train_ds,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=num_workers,
        #     pin_memory=(num_workers == 0),
        #     persistent_workers=num_workers > 0,
        #     prefetch_factor=4 if num_workers > 0 else None,
        #     collate_fn=custom_collate,
        # )

        # val_loader = torch.utils.data.DataLoader(
        #     train_ds,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=num_workers,
        #     pin_memory=(num_workers == 0),
        #     persistent_workers=num_workers > 0,
        #     prefetch_factor=4 if num_workers > 0 else None,
        #     collate_fn=custom_collate,
        # )

        yield fold, train_loader, val_loader


def make_cv_loaders(data_root, **kwargs):
    samples = collect_img_paths_with_labels(data_root)
    yield from make_cv_loaders_from_samples(samples, **kwargs)
