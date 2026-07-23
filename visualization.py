import os
from collections import defaultdict

import cv2
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from scipy.ndimage import gaussian_filter


def _set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2,
    })


def _group_by_image(paths):
    groups = defaultdict(list)
    for i, path in enumerate(paths):
        groups[path].append(i)
    return groups


def _save_fig(path, dpi=300):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def build_heatmap(H, W, patches_list, sigma=2):
    _set_plot_style()
    heatmap = np.zeros((H, W), dtype=float)
    countmap = np.zeros((H, W), dtype=float)

    for item in patches_list:
        x1, y1, x2, y2 = map(int, item["coords"])
        prob = float(item["prob_pred"])
        heatmap[y1:y2, x1:x2] += prob
        countmap[y1:y2, x1:x2] += 1

    mask = countmap > 0
    heatmap[mask] /= countmap[mask]

    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap


def plot_misclassified_with_heatmap(results_path, misclassified_list, sigma=2, alpha=0.3, postfix=None):
    _set_plot_style()

    img_to_patches = defaultdict(list)
    for item in misclassified_list:
        img_to_patches[item["img_path"]].append(item)

    for img_path, patches_list in img_to_patches.items():
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W, _ = img.shape
        heatmap = build_heatmap(H, W, patches_list, sigma=sigma)
        base = os.path.splitext(os.path.basename(img_path))[0]
        prefix = f"secondary_{base}" if postfix else base

        # Combined (image + heatmap + boxes)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        im = ax.imshow(heatmap, cmap="inferno", alpha=alpha, origin="upper", vmin=0, vmax=1)
        for item in patches_list:
            x1, y1, x2, y2 = map(int, item["coords"])
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="cyan", facecolor="none")
            ax.add_patch(rect)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence Score")
        _save_fig(os.path.join(results_path, f"combined_{prefix}.png"))

        # Boxes only
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        for item in patches_list:
            x1, y1, x2, y2 = map(int, item["coords"])
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
        ax.axis("off")
        _save_fig(os.path.join(results_path, f"boxes_{prefix}.png"))

        # Heatmap only
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(heatmap, cmap="inferno", origin="upper", vmin=0, vmax=1)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence Score")
        _save_fig(os.path.join(results_path, f"heatmap_{prefix}.png"))

    print(f"Saved {len(img_to_patches)} images (combined + boxes + heatmap).")


def plot_misclassified(results_path, misclassified_list):
    _set_plot_style()

    img_to_patches = defaultdict(list)
    for item in misclassified_list:
        img_to_patches[item["img_path"]].append(item)

    for img_path, patches_list in img_to_patches.items():
        img = np.array(Image.open(img_path).convert("RGB"))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.axis("off")

        for item in patches_list:
            x1, y1, x2, y2 = map(int, item["coords"])
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

        base = os.path.splitext(os.path.basename(img_path))[0]
        _save_fig(os.path.join(results_path, f"missed_boxes_{base}.png"))


def create_heatmaps_per_image(results_path, all_eval_paths, coords, probs, sigma=0):
    _set_plot_style()
    img_to_indices = _group_by_image(all_eval_paths)

    for img_path, indices in img_to_indices.items():
        sample_img = Image.open(img_path)
        W, H = sample_img.size

        heatmap = np.zeros((H, W), dtype=float)
        countmap = np.zeros((H, W), dtype=float)

        for idx in indices:
            x1, y1, x2, y2 = map(int, coords[idx])
            heatmap[y1:y2, x1:x2] += probs[idx]
            countmap[y1:y2, x1:x2] += 1

        mask = countmap > 0
        heatmap[mask] /= countmap[mask]

        base = os.path.splitext(os.path.basename(img_path))[0]
        np.save(os.path.join(results_path, f"heatmap_{base}.npy"), heatmap)

        masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
        if sigma > 0:
            masked_heatmap = gaussian_filter(heatmap, sigma=sigma)

        vmin = masked_heatmap.min()
        vmax = masked_heatmap.max()

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(masked_heatmap, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence")
        _save_fig(os.path.join(results_path, f"confidence_{base}.png"))

    print(f"Saved {len(img_to_indices)} heatmaps.")


def create_misclassification_density_maps(results_path, all_eval_paths, coords, preds, trues):
    _set_plot_style()
    img_to_indices = _group_by_image(all_eval_paths)

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

        base = os.path.splitext(os.path.basename(img_path))[0]
        np.save(os.path.join(results_path, f"error_map_{base}.npy"), error_map)

        error_map = gaussian_filter(error_map, sigma=3)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(error_map, cmap="magma", origin="upper")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Error Density")
        _save_fig(os.path.join(results_path, f"error_density_{base}.png"))


def create_per_class_error_maps(results_path, all_eval_paths, coords, preds, trues, target_class):
    _set_plot_style()
    img_to_indices = _group_by_image(all_eval_paths)

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

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(class_error_map, cmap="inferno", origin="upper")
        fig.colorbar(im, label="Class Error Density")

        base = os.path.splitext(os.path.basename(img_path))[0]
        _save_fig(os.path.join(results_path, f"class_{target_class}_error_{base}.png"))


def create_uncertainty_maps(results_path, all_eval_paths, coords, prob_vectors, misclassified_list):
    _set_plot_style()

    img_to_indices = _group_by_image(all_eval_paths)

    mis_by_img = defaultdict(list)
    for item in misclassified_list:
        mis_by_img[item["img_path"]].append(item)

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

        base = os.path.splitext(os.path.basename(img_path))[0]
        np.save(os.path.join(results_path, f"uncertainty_map_{base}.npy"), uncertainty_map)
        uncertainty_map = gaussian_filter(uncertainty_map, sigma=2)

        # Pure uncertainty map
        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(uncertainty_map, cmap=sns.color_palette("mako", as_cmap=True), origin="upper")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Prediction Entropy")
        ax.set_title("Uncertainty Map")
        plt.tight_layout()
        _save_fig(os.path.join(results_path, f"uncertainty_{base}.png"), dpi=600)

        # Overlay: grayscale + uncertainty heatmap + misclassified boxes
        img_gray = np.array(img.convert("L"))
        unc_norm = uncertainty_map - uncertainty_map.min()
        if unc_norm.max() > 0:
            unc_norm = unc_norm / unc_norm.max()

        heatmap_color = cv2.applyColorMap((unc_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        img_gray_rgb = np.stack([img_gray] * 3, axis=-1)
        img_gray_bgr = cv2.cvtColor(img_gray_rgb, cv2.COLOR_RGB2BGR)

        alpha = unc_norm[..., None]
        img_with_uncertainty = cv2.convertScaleAbs(
            img_gray_bgr * (1 - alpha) + heatmap_color * alpha
        )
        img_with_uncertainty = np.clip(img_with_uncertainty, 0, 255).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_with_uncertainty, origin="upper")

        for item in mis_by_img.get(img_path, []):
            x1, y1, x2, y2 = map(int, item["coords"])
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

        legend_patch = mpatches.Patch(edgecolor="red", facecolor="none",
                                      linewidth=2, label="Misclassified Patch")
        ax.legend(handles=[legend_patch], loc="upper right", frameon=True)

        norm = mcolors.Normalize(vmin=unc_norm.min(), vmax=unc_norm.max())
        sm = cm.ScalarMappable(norm=norm, cmap="magma")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Prediction Uncertainty (Entropy)")
        ax.axis("off")
        plt.tight_layout()
        _save_fig(os.path.join(results_path, f"uncertainty_overlay_{base}.png"), dpi=600)
