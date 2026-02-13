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
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def plot_misclassified(misclassified_list):
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

        for item in patches_list:
            x1, y1, x2, y2 = map(int, item["coords"])
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

        #     # Fill heatmap for this patch
        #     prob = float(item["prob_pred"])
        #     heatmap[y1:y2, x1:x2] = prob

        # # --- Plot heatmap ---
        # im = ax_heat.imshow(heatmap, cmap='inferno', origin='upper')
        # ax_heat.axis("off")
        # ax_heat.set_title("Confidence Heatmap")
        # cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        # cbar.set_label("Confidence Score")


        base = os.path.splitext(os.path.basename(img_path))[0]
        plt.savefig(f"missed_boxes_{base}.png", dpi=300)
        plt.close(fig)
        

def create_heatmaps_per_image(all_eval_paths, coords, probs, sigma=2):
    """
    Creates one confidence heatmap per evaluated image.
    all_eval_paths, coords, probs must be aligned.
    """

    # -----------------------------
    # Group indices per image
    # -----------------------------
    img_to_indices = defaultdict(list)

    for i, img_path in enumerate(all_eval_paths):
        img_to_indices[img_path].append(i)

    # -----------------------------
    # Build heatmap per image
    # -----------------------------
    for img_path, indices in img_to_indices.items():

        # Load image to get correct size
        sample_img = Image.open(img_path)
        W, H = sample_img.size

        heatmap = np.zeros((H, W), dtype=float)
        countmap = np.zeros((H, W), dtype=float)

        # Accumulate patch confidences
        for idx in indices:
            x1, y1, x2, y2 = map(int, coords[idx])
            
            confidence = probs[idx]

            heatmap[y1:y2, x1:x2] += confidence
            countmap[y1:y2, x1:x2] += 1

        # Normalize overlapping areas
        mask = countmap > 0
        heatmap[mask] /= countmap[mask]

        # Optional smoothing
        if sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=sigma)

        # -----------------------------
        # Plot
        # -----------------------------
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap="inferno", origin="upper")
        plt.colorbar(label="Confidence")
        plt.title(f"Confidence Heatmap\n{os.path.basename(img_path)}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        plt.savefig(f"confidence_{base}.png", dpi=300)
        plt.close()

    print(f"Saved {len(img_to_indices)} heatmaps.")


def create_misclassification_density_maps(all_eval_paths, coords, preds, trues):
    """
    Creates per-image misclassification density maps.
    """

    from collections import defaultdict
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    img_to_indices = defaultdict(list)
    for i, path in enumerate(all_eval_paths):
        img_to_indices[path].append(i)

    for img_path, indices in img_to_indices.items():

        img = Image.open(img_path)
        W, H = img.size

        error_map = np.zeros((H, W), dtype=float)
        count_map = np.zeros((H, W), dtype=float)

        for idx in indices:
            x1, y1, x2, y2 = map(int, coords[idx])
            is_error = int(preds[idx] != trues[idx])

            error_map[y1:y2, x1:x2] += is_error
            count_map[y1:y2, x1:x2] += 1

        mask = count_map > 0
        error_map[mask] /= count_map[mask]

        error_map = gaussian_filter(error_map, sigma=2)

        plt.figure(figsize=(6, 6))
        plt.imshow(error_map, cmap="magma", origin="upper")
        plt.colorbar(label="Misclassification Density")
        plt.title(f"Error Density\n{os.path.basename(img_path)}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        plt.savefig(f"error_density_{base}.png", dpi=300)
        plt.close()

def create_per_class_error_maps(all_eval_paths, coords, preds, trues, target_class):
    """
    target_class: int label to analyze
    """

    from collections import defaultdict
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    img_to_indices = defaultdict(list)
    for i, path in enumerate(all_eval_paths):
        img_to_indices[path].append(i)

    for img_path, indices in img_to_indices.items():

        img = Image.open(img_path)
        W, H = img.size

        class_error_map = np.zeros((H, W), dtype=float)
        count_map = np.zeros((H, W), dtype=float)

        for idx in indices:
            if trues[idx] == target_class:

                x1, y1, x2, y2 = map(int, coords[idx])
                is_error = int(preds[idx] != trues[idx])

                class_error_map[y1:y2, x1:x2] += is_error
                count_map[y1:y2, x1:x2] += 1

        mask = count_map > 0
        class_error_map[mask] /= count_map[mask]

        class_error_map = gaussian_filter(class_error_map, sigma=2)

        plt.figure(figsize=(6, 6))
        plt.imshow(class_error_map, cmap="inferno", origin="upper")
        plt.colorbar(label="Class Error Density")
        plt.title(f"Class {target_class} Error Map\n{os.path.basename(img_path)}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        plt.savefig(f"class_{target_class}_error_{base}.png", dpi=300)
        plt.close()

def create_uncertainty_maps(all_eval_paths, coords, prob_vectors):
    """
    prob_vectors: list of softmax vectors per patch (shape: [N, num_classes])
    """

    from collections import defaultdict
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    img_to_indices = defaultdict(list)
    for i, path in enumerate(all_eval_paths):
        img_to_indices[path].append(i)

    for img_path, indices in img_to_indices.items():

        img = Image.open(img_path)
        W, H = img.size

        uncertainty_map = np.zeros((H, W), dtype=float)
        count_map = np.zeros((H, W), dtype=float)

        for idx in indices:
            x1, y1, x2, y2 = map(int, coords[idx])

            probs = np.array(prob_vectors[idx])
            entropy = -np.sum(probs * np.log(probs + 1e-12))

            uncertainty_map[y1:y2, x1:x2] += entropy
            count_map[y1:y2, x1:x2] += 1

        mask = count_map > 0
        uncertainty_map[mask] /= count_map[mask]

        uncertainty_map = gaussian_filter(uncertainty_map, sigma=2)

        plt.figure(figsize=(6, 6))
        plt.imshow(uncertainty_map, cmap="viridis", origin="upper")
        plt.colorbar(label="Prediction Entropy")
        plt.title(f"Uncertainty Map\n{os.path.basename(img_path)}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        plt.savefig(f"uncertainty_{base}.png", dpi=300)
        plt.close()

    
def main():
    # lbp_settings=[(16, 2), (24,3), (32, 4)]
    lbp_settings=None
    print("Hello from bainite-martensite!")
    train_loader, val_loader, test_loader = make_train_val_test_loaders(
        data_root,
        batch_size=128,
        patch_size=256,
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

    missclassified, coords, probs, all_eval_paths, preds, trues, prob_vectors = evaluate_and_visualize_single_head(
        model=model,
        loader=val_loader,
        device=device,
        max_show=len(val_loader.dataset)
    )



    # 1️⃣ Confidence heatmaps
    create_heatmaps_per_image(all_eval_paths, coords, probs, sigma=2)

    # 2️⃣ Misclassified patches
    plot_misclassified(missclassified)

    # 3️⃣ Misclassification density
    create_misclassification_density_maps(all_eval_paths, coords, preds, trues)

    # 4️⃣ Per-class error maps (example target_class = 0)
    create_per_class_error_maps(all_eval_paths, coords, preds, trues, target_class=0)

    # 5️⃣ Uncertainty maps
    create_uncertainty_maps(all_eval_paths, coords, prob_vectors)

if __name__ == "__main__":
    main()
