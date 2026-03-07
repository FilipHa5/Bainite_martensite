import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from models import normalized_histogram
from PIL import Image
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns

@torch.no_grad()
def evaluate_and_visualize_single_head(
    result_path,
    model,
    loader,
    device,
    max_show=16,
    class_names=None,  # optional list of class names,
    secondary_model=None,
    min_confidence_threshold=None,
    n_bins=16,
    min_margin=0.1
):
    model.eval()

    all_true = []
    all_pred = []
    all_dominant_probs = []
    all_coords = []
    all_prob_vectors = []
    misclassified = []
    all_eval_paths = []
    hybrid_pred=[]
    secondary_pred_correct_ctr=0
    secondary_pred_missed_ctr=0
    total_hybrid_correct=0

    for batch in loader:
        rgb = batch["rgb"].to(device)
        # lbp=batch["lbp"].to(device)
        labels = batch["label"].to(device)
        coords = batch["coords"].to(device)
        img_paths = batch["img_path"]

        lbp_enabled = batch.get("lbp", None)
        if lbp_enabled is not None:
            lbp = batch["lbp"].to(device, non_blocking=True)
            logits = model(rgb, lbp)
        else:
            logits = model(rgb)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        for i in range(rgb.size(0)):
            all_true.append(labels[i].item())
            all_pred.append(preds[i].item())
            all_dominant_probs.append(probs[i][preds[i]].item())
            all_coords.append(coords[i].cpu().tolist())
            all_prob_vectors.append(probs[i].cpu().numpy())
            all_eval_paths.append(img_paths[i])

            sorted_probs = torch.sort(probs[i], descending=True).values
            margin = sorted_probs[0] - sorted_probs[1]  # conf1 - conf2

            if secondary_model:
                if probs[i][preds[i]].item() < min_confidence_threshold:# or margin < min_margin and margin is not None:
                    rgb_hist = normalized_histogram(rgb[i].cpu(), bins=n_bins)
                    if lbp_enabled is not None:
                        lbp_hist = normalized_histogram(lbp[i].cpu(), bins=n_bins)
                    else:
                        lbp_hist = np.zeros(n_bins, dtype=np.float32)

                    features = np.concatenate([rgb_hist, lbp_hist])

                    pred_dt = secondary_model.predict(features.reshape(1,-1))
                    hybrid_pred.append(pred_dt[0])
                    print(f"Confidence: {probs[i][preds[i]].item()}", end=";")
                    if pred_dt[0] != labels[i].item():
                        secondary_pred_missed_ctr+=1
                        print(f"DT Fucked up. Pred: {pred_dt[0]}, true: {labels[i].item()}")
                    else:
                        secondary_pred_correct_ctr+=1
                        total_hybrid_correct+=1
                        print(f"DT worked well. Pred: {pred_dt[0]}, true: {labels[i].item()}")
                        
                else:
                    hybrid_pred.append(preds[i].item())
                    
            if preds[i] != labels[i] and len(misclassified) < max_show:
                misclassified.append({
                    "image": rgb[i].cpu(),
                    "img_path": img_paths[i],
                    "true": labels[i].item(),
                    "pred": preds[i].item(),
                    "prob_pred": probs[i][preds[i]].item(),
                    "coords": coords[i].cpu().tolist()
                })

    # ---- Compute accuracy ----
    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)
    hybrid_pred_np = np.array(hybrid_pred)
    test_accuracy = (all_true_np == all_pred_np).mean()
    
    hybrid_accuracy = (all_true_np == hybrid_pred_np).mean()
    
    # Generate classification report
    report_dict = classification_report(
        all_true,
        hybrid_pred_np,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    with open(os.path.join(result_path, "classification_report.txt"), "w") as f:
        f.write(report_df.to_string(float_format="%.4f"))
    report_df.to_csv(os.path.join(result_path, "classification_report.csv"), float_format="%.4f")
    

    conf_matrix = confusion_matrix(all_true, hybrid_pred_np)
    if class_names is None:
        class_names = [str(i) for i in range(conf_matrix.shape[0])]
    np.save(os.path.join(result_path, "confusion_matrix.npy"), conf_matrix)
    cm_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(result_path, "confusion_matrix.csv"))

    cm_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".3f",
        cmap="Greys",
        cbar=True,
        square=True,
        linewidths=0.8,
        linecolor="black",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    ax.set_title("Normalized Confusion Matrix", pad=15)

    # Improve tick formatting
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Remove unnecessary spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(result_path,"confusion_matrix_plot.pdf"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(result_path,"confusion_matrix_plot.png"), dpi=600, bbox_inches="tight")

    plt.close()

    return (
        misclassified,
        all_coords,
        all_dominant_probs,
        all_eval_paths,
        all_pred,
        all_true,
        all_prob_vectors,
        report_dict,
        conf_matrix,
        test_accuracy,
        hybrid_accuracy
    )

def perform_all_patches_corrections(all_coords, all_dominant_probs, all_eval_paths, all_pred, all_true):
    for coords, prob, img_path, pred, true in zip(all_coords, all_dominant_probs, all_eval_paths, all_pred, all_true):
        print(prob)

def build_heatmap(H, W, patches_list, sigma=2):
    """
    Build normalized heatmap from patch confidence scores.
    Handles overlapping regions properly.
    """
    heatmap = np.zeros((H, W), dtype=float)
    countmap = np.zeros((H, W), dtype=float)

    for item in patches_list:
        x1, y1, x2, y2 = map(int, item["coords"])
        prob = float(item["prob_pred"])

        heatmap[y1:y2, x1:x2] += prob
        countmap[y1:y2, x1:x2] += 1

    # Normalize overlaps
    mask = countmap > 0
    heatmap[mask] /= countmap[mask]

    # Optional smoothing
    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap
def plot_misclassified_with_heatmap(results_path, misclassified_list, sigma=2, alpha=0.5):

    img_to_patches = defaultdict(list)
    for item in misclassified_list:
        img_to_patches[item["img_path"]].append(item)

    for img_path, patches_list in img_to_patches.items():

        img = np.array(Image.open(img_path).convert("RGB"))
        H, W, _ = img.shape
        heatmap = build_heatmap(H, W, patches_list, sigma=sigma)

        base = os.path.splitext(os.path.basename(img_path))[0]

        # =====================================================
        # 1️⃣ COMBINED (image + heatmap + boxes)
        # =====================================================
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        im = ax.imshow(
            heatmap,
            cmap="inferno",
            alpha=alpha,
            origin="upper",
            vmin=0,
            vmax=1
        )

        for item in patches_list:
            x1, y1, x2, y2 = map(int, item["coords"])
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='cyan',
                facecolor='none'
            )
            ax.add_patch(rect)

        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence Score")

        combined_path = os.path.join(
            results_path, f"combined_{base}.png"
        )
        plt.savefig(combined_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # =====================================================
        # 2️⃣ IMAGE + BOXES ONLY
        # =====================================================
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        for item in patches_list:
            x1, y1, x2, y2 = map(int, item["coords"])
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

        ax.axis("off")

        boxes_path = os.path.join(
            results_path, f"boxes_{base}.png"
        )
        plt.savefig(boxes_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # =====================================================
        # 3️⃣ HEATMAP ONLY
        # =====================================================
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            heatmap,
            cmap="inferno",
            origin="upper",
            vmin=0,
            vmax=1
        )
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence Score")

        heatmap_path = os.path.join(
            results_path, f"heatmap_{base}.png"
        )
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(img_to_patches)} images (combined + boxes + heatmap).")

def plot_misclassified(results_path, misclassified_list):
    """"
    Plot misclassified patches on original images with a separate spatial heatmap
    of confidence scores using Inferno colormap and color scale.

    misclassified_list: list of dicts, each must contain:
        - "img_path": path to original image
        - "coords": (x1, y1, x2, y2)
        - "true": true label
        - "pred": predicted label
        - "prob_pred": confidence score (0-1)
    """

    # n_bins = clf.bins_used

    # Group patches by image
    img_to_patches = defaultdict(list)
    for item in misclassified_list:
        img_to_patches[item["img_path"]].append(item)

    for img_path, patches_list in img_to_patches.items():
        # Load original image
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W, _ = img.shape

        # Create figure with 2 subplots: image and heatmap
        # fig, (ax_img, ax_heat) = plt.subplots(1, 2, figsize=(16, 8))
        fig, ax_img = plt.subplots(1, 1, figsize=(8, 8))

        # --- Plot original image with rectangles ---
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(f"Misclassified Patches - {item["img_path"]}")

        for item in patches_list:
            # rgb_hist = normalized_histogram(item["image"], bins=n_bins)
            # if "lbp" in item and item["lbp"] is not None:
            #     lbp_hist = normalized_histogram(item["lbp"], bins=n_bins)
            # else:
            #     lbp_hist = np.zeros(n_bins, dtype=np.float32)

            # features = np.concatenate([rgb_hist, lbp_hist])
            # clf_pred = clf.predict(features.reshape(1, -1))
            # print(f"DT Prediction: {clf_pred}, NN Prediction: {item['pred']}, True: {item['true']}")
            # if clf_pred == item['true'] and item['pred'] != item['true']:
            #     print("DT got it right, but NN not")
            # if clf_pred != item['true'] and item['pred'] != item['true']:
            #     print("Thats a ahrd one, both algs made a msitake")
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
        path_to_save = os.path.join(results_path, f"missed_boxes_{base}.png")
        plt.savefig(path_to_save, dpi=300)
        plt.close(fig)
        

def create_heatmaps_per_image(results_path, all_eval_paths, coords, probs, sigma=2):
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
        path_to_save = os.path.join(results_path, f"confidence_{base}.png")
        plt.savefig(path_to_save, dpi=300)
        plt.close()

    print(f"Saved {len(img_to_indices)} heatmaps.")


def create_misclassification_density_maps(results_path, all_eval_paths, coords, preds, trues,):
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
        
        
        path_to_save = os.path.join(results_path, f"error_density_{base}.png")
        plt.savefig(path_to_save, dpi=300)
        plt.close()

def create_per_class_error_maps(results_path, all_eval_paths, coords, preds, trues, target_class):
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
        path_to_save = os.path.join(results_path, f"class_{target_class}_error_{base}.png")
        plt.savefig(path_to_save, dpi=300)
        plt.close()

def create_uncertainty_maps(results_path, all_eval_paths, coords, prob_vectors):
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
        base = os.path.splitext(os.path.basename(img_path))[0]
        path_to_save = os.path.join(results_path, f"uncertainty_{base}.png")
        plt.close() 