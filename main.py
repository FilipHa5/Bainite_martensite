import torch
import torch.nn as nn
import torchvision.models as models

from loaders import make_train_val_test_loaders, make_cv_loaders
from network import MicrostructureNet
from dataset import MicrostructurePatchDataset
from evaluate import evaluate_and_visualize_single_head
from train import train, train_single_val

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import os

data_root = os.path.join("images")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import numpy as np
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from PIL import Image

def plot_misclassified_with_heatmap(misclassified_list):
    """
    Plot misclassified patches on original images with a separate spatial heatmap
    of confidence scores using Inferno colormap and color scale.

    misclassified_list: list of dicts, each must contain:
        - "img_path": path to original image
        - "coords": (x1, y1, x2, y2)
        - "true": true label
        - "pred": predicted label
        - "prob_pred": confidence score (0-1)
    """

    # Group patches by image
    img_to_patches = defaultdict(list)
    for item in misclassified_list:
        img_to_patches[item["img_path"]].append(item)

    for img_path, patches_list in img_to_patches.items():
        # Load original image
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W, _ = img.shape

        # Create figure with 2 subplots: image and heatmap
        fig, (ax_img, ax_heat) = plt.subplots(1, 2, figsize=(16, 8))

        # --- Plot original image with rectangles ---
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title("Misclassified Patches")

        # --- Prepare heatmap array ---
        heatmap = np.zeros((H, W))

        for item in patches_list:
            x1, y1, x2, y2 = item["coords"]
            width = x2 - x1
            height = y2 - y1

            # Draw rectangle on original image
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax_img.add_patch(rect)

            # Add label
            ax_img.text(
                x1,
                y1 - 5,
                f"T:{item['true']} P:{item['pred']}",
                color='yellow',
                fontsize=8,
                bbox=dict(facecolor='black', alpha=0.6)
            )

            # Fill heatmap for this patch
            prob = float(item["prob_pred"])
            heatmap[y1:y2, x1:x2] = prob

        # --- Plot heatmap ---
        im = ax_heat.imshow(heatmap, cmap='inferno', origin='upper')
        ax_heat.axis("off")
        ax_heat.set_title("Confidence Heatmap")
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence Score")

        plt.tight_layout()
        plt.show()


def main():
    lbp_settings=[(1, 8)]
    # lbp_settings=None
    print("Hello from bainite-martensite!")
    train_loader, val_loader, test_loader = make_train_val_test_loaders(
        data_root,
        batch_size=128,
        patch_size=128,
        stride=64,
        lbp_settings=lbp_settings,
        val_frac=0.15,
        test_frac=0.15,
        num_workers=0, # Windows Jupyter safe - multiprocessing issue
        random_state=42
    )


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # ------------------ Model ------------------
    model = MicrostructureNet(lbp_settings=lbp_settings, freeze_backbone=True).to(device)

    # --------------- Loss + Optimizer-----------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    model, history = train_single_val(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        num_epochs=50,
        patience=7,
    )

    missclassified, coords, probs = evaluate_and_visualize_single_head(
        model=model,
        loader=val_loader,
        device=device,
        max_show=32
    )

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    # # Canvas size (adjust as needed)
    # H, W = 100, 100

    # # Heatmap accumulation
    # heatmap = np.zeros((H, W), dtype=float)
    # countmap = np.zeros((H, W), dtype=float)

    # Accumulate values
    # for (x1, y1, x2, y2), confidence in zip(coords, probs):
    #     heatmap[y1:y2, x1:x2] += confidence
    #     countmap[y1:y2, x1:x2] += 1

    # # Avoid division by zero
    # mask = countmap > 0
    # heatmap[mask] /= countmap[mask]

    # # Smooth overlaps
    # heatmap = gaussian_filter(heatmap, sigma=2)

    # # Plot
    # plt.figure(figsize=(6, 6))
    # plt.imshow(heatmap, cmap="inferno", origin="lower")
    # plt.colorbar(label="Confidencealue intensity")
    # plt.title("Confidence per Box Heatmap")
    # plt.show()

    plot_misclassified_with_heatmap(missclassified)

if __name__ == "__main__":
    main()
