import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

from datasets import MicrostructurePatchDataset
from .img_utils import collect_img_paths_with_labels
from .collate import custom_collate

def make_train_val_test_loaders(
    data_root,
    batch_size=128,
    patch_size=128,
    stride=64,
    lbp_settings=[(1, 8), (3, 16)],
    val_frac=0.15,
    test_frac=0.15,
    num_workers=0, # Windows Jupyter safe - multiprocessing issue
    random_state=42
):
    img_paths_label = collect_img_paths_with_labels(data_root)

    print(f"Collected {len(img_paths_label)} images")

    # Stratify by family (0=UB, 1=Lath-type)
    y_family = np.array([s[1] for s in img_paths_label])

    # -----------------------
    # Split off TEST set
    # -----------------------
    sss_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_frac,
        random_state=random_state
    )

    trainval_idx, test_idx = next(sss_test.split(img_paths_label, y_family))

    trainval_samples = [img_paths_label[i] for i in trainval_idx]
    test_samples     = [img_paths_label[i] for i in test_idx]

    y_trainval = y_family[trainval_idx]

    # -----------------------
    # Split TRAIN / VAL
    # -----------------------
    val_size_rel = val_frac / (1.0 - test_frac)

    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size_rel,
        random_state=random_state
    )

    train_idx, val_idx = next(sss_val.split(trainval_samples, y_trainval))

    train_samples = [trainval_samples[i] for i in train_idx]
    val_samples   = [trainval_samples[i] for i in val_idx]

    # -----------------------
    # Datasets
    # -----------------------
    train_ds = MicrostructurePatchDataset(
        train_samples,
        patch_size=patch_size,
        stride=stride,
        augment=True,
        lbp_settings=lbp_settings
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
        lbp_settings=lbp_settings
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
        collate_fn=custom_collate
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers == 0),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=custom_collate
        
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers == 0),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=custom_collate
        
    )


    print("Train family distribution:", np.bincount(y_family[train_idx]))
    print("Val family distribution:", np.bincount(y_family[val_idx]))
    print("Test family distribution:", np.bincount(y_family[test_idx]))
    return train_loader, val_loader, test_loader


def make_cv_loaders(
    data_root,
    batch_size=128,
    n_splits=5,
    patch_size=256,
    stride=64,
    lbp_settings=[(1, 8), (3, 16)]
):
    img_paths_label = collect_img_paths_with_labels(data_root)
    # Stratify by family (0=UB, 1=Lath-type) so each fold has both classes
    y_family = np.array([s[1] for s in img_paths_label])
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(img_paths_label, y_family)):
        train_samples = [img_paths_label[i] for i in train_idx]
        val_samples   = [img_paths_label[i] for i in val_idx]

        train_ds = MicrostructurePatchDataset(
            train_samples,
            patch_size=patch_size,
            stride=stride,
            augment=True,
            lbp_settings=lbp_settings
        )

        val_ds = MicrostructurePatchDataset(
            val_samples,
            patch_size=patch_size,
            stride=stride,
            augment=False,
            lbp_settings=lbp_settings
        )

        # num_workers=0 on Windows avoids multiprocessing hang in Jupyter (first batch never arrives)
        num_workers = 0
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(num_workers == 0),
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None
        )

        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(num_workers == 0),
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None
        )

        yield fold, train_loader, val_loader