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
    class_names=None,
    secondary_model=None,
    secondary_type="xgb",
    min_confidence_threshold=0.75,
    n_bins=16,
    confidence_margin=0.15,
    min_secondary_confidence=0.75,
):
    model.eval()

    if secondary_model and secondary_type == "cnn":
        secondary_model.eval()

    # ------------------------------------------------------------------ #
    #  Secondary forward pass (XGBoost or CNN)                           #
    # ------------------------------------------------------------------ #
    def secondary_forward(rgb_tensor, lbp_tensor):
        if secondary_model is None:
            return {"pred": None, "conf": None, "probs": None}

        if secondary_type == "xgb":
            rgb_hist = normalized_histogram(rgb_tensor.cpu(), bins=n_bins)
            lbp_hist = (
                normalized_histogram(lbp_tensor.cpu(), bins=n_bins)
                if lbp_tensor is not None else np.zeros(n_bins, dtype=np.float32)
            )
            features = np.concatenate([rgb_hist, lbp_hist])

            dt_pred = secondary_model.predict(features.reshape(1, -1))[0]
            dt_probs = secondary_model.predict_proba(features.reshape(1, -1))[0]
            dt_conf = dt_probs[dt_pred]

            return {"pred": dt_pred, "conf": dt_conf, "probs": dt_probs}

        elif secondary_type == "cnn":
            if lbp_tensor is not None:
                logits = secondary_model(rgb_tensor.unsqueeze(0), lbp_tensor.unsqueeze(0))
            else:
                logits = secondary_model(rgb_tensor.unsqueeze(0))

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred])

            return {"pred": pred, "conf": conf, "probs": probs}

        else:
            raise ValueError(f"Unknown secondary_type: {secondary_type}")

    # ------------------------------------------------------------------ #
    #  Hybrid gating decision (single source of truth)                   #
    # ------------------------------------------------------------------ #
    def hybrid_decision(nn_pred, nn_conf, dt_pred, dt_conf, adaptive_threshold):
        """Decide whether the secondary classifier should override the NN.

        Returns:
            use_xgb : bool  – True when the secondary prediction is accepted
            decision : str  – human-readable reason for the decision
        """
        if dt_pred is None:
            return False, "No secondary model"

        if nn_conf >= adaptive_threshold:
            return False, "High NN confidence"

        if nn_pred == dt_pred:
            return False, "Agreement"

        if dt_conf < min_secondary_confidence:
            return False, "Low XGB confidence"

        delta = dt_conf - nn_conf
        if delta < confidence_margin:
            return False, "Small confidence margin"

        return True, "Override"

    # ------------------------------------------------------------------ #
    #  Global storage                                                     #
    # ------------------------------------------------------------------ #
    all_true = []
    all_pred = []
    all_dominant_probs = []
    all_coords = []
    all_prob_vectors = []
    all_eval_paths = []
    all_margins = []
    all_dt_prob_vectors = []

    all_dt_pred = []
    all_dt_confidence = []

    all_images = []
    all_img_paths = []
    all_coords_list = []

    misclassified_nn = []
    misclassified_secondary = []

    secondary_info = []

    # ------------------------------------------------------------------ #
    #  Main evaluation loop                                               #
    # ------------------------------------------------------------------ #
    for batch in loader:
        rgb = batch["rgb"].to(device)
        labels = batch["label"].to(device)
        coords = batch["coords"].to(device)
        img_paths = batch["img_path"]
        lbp_enabled = batch.get("lbp", None)

        if lbp_enabled is not None:
            lbp = batch["lbp"].to(device)
            logits = model(rgb, lbp)
        else:
            lbp = None
            logits = model(rgb)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        for i in range(rgb.size(0)):
            true_label = labels[i].item()
            nn_pred = preds[i].item()
            confidence = probs[i][nn_pred].item()

            sorted_probs = torch.sort(probs[i], descending=True).values
            margin = (sorted_probs[0] - sorted_probs[1]).item()

            all_true.append(true_label)
            all_pred.append(nn_pred)
            all_dominant_probs.append(confidence)
            all_coords.append(coords[i].cpu().tolist())
            all_prob_vectors.append(probs[i].cpu().numpy())
            all_eval_paths.append(img_paths[i])
            all_margins.append(margin)

            all_images.append(rgb[i].cpu())
            all_img_paths.append(img_paths[i])
            all_coords_list.append(coords[i].cpu().tolist())

            # secondary model forward pass
            sec_out = secondary_forward(
                rgb[i],
                lbp[i] if lbp is not None else None
            )

            dt_pred = sec_out["pred"]
            dt_conf = sec_out["conf"]
            dt_probs = sec_out["probs"]

            if secondary_model:
                all_dt_pred.append(dt_pred)
                all_dt_confidence.append(dt_conf)
                all_dt_prob_vectors.append(dt_probs)

            # collect NN misclassifications for visualisation
            if nn_pred != true_label and len(misclassified_nn) < max_show:
                misclassified_nn.append({
                    "image": rgb[i].cpu(),
                    "img_path": img_paths[i],
                    "true": true_label,
                    "pred": nn_pred,
                    "prob_pred": confidence,
                    "coords": coords[i].cpu().tolist()
                })

            secondary_info.append({
                "nn_pred": nn_pred,
                "nn_conf": confidence,
                "dt_pred": dt_pred,
                "dt_conf": dt_conf,
                "true": true_label
            })

    # ------------------------------------------------------------------ #
    #  Hybrid classification                                              #
    # ------------------------------------------------------------------ #
    adaptive_threshold = np.clip(
        np.percentile(all_dominant_probs, 10),
        0.75,
        0.90
    )

    hybrid_pred = []

    successful_overrides = 0
    failed_overrides = 0
    secondary_used = 0
    reject_agreement = 0
    reject_low_confidence = 0
    reject_small_margin = 0
    override_deltas = []

    for info in secondary_info:
        use_xgb, decision = hybrid_decision(
            info["nn_pred"], info["nn_conf"],
            info["dt_pred"], info["dt_conf"],
            adaptive_threshold
        )

        if use_xgb:
            hybrid_pred.append(info["dt_pred"])
            secondary_used += 1
            override_deltas.append(info["dt_conf"] - info["nn_conf"])

            if info["dt_pred"] == info["true"]:
                successful_overrides += 1
            else:
                failed_overrides += 1
        else:
            hybrid_pred.append(info["nn_pred"])

            if info["nn_conf"] < adaptive_threshold:
                if decision == "Agreement":
                    reject_agreement += 1
                elif decision == "Low XGB confidence":
                    reject_low_confidence += 1
                elif decision == "Small confidence margin":
                    reject_small_margin += 1

    # ------------------------------------------------------------------ #
    #  Accuracy                                                           #
    # ------------------------------------------------------------------ #
    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)
    hybrid_pred_np = np.array(hybrid_pred)

    test_accuracy = (all_pred_np == all_true_np).mean()
    hybrid_accuracy = (hybrid_pred_np == all_true_np).mean()

    # ------------------------------------------------------------------ #
    #  Classification reports                                             #
    # ------------------------------------------------------------------ #
    if secondary_model and len(all_dt_pred) > 0:
        all_dt_pred_np = np.array(all_dt_pred)

        report_df_dt = pd.DataFrame(
            classification_report(
                all_true,
                all_dt_pred_np,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
        ).transpose()

        report_df_dt.to_csv(os.path.join(result_path, "classification_report_dt.csv"))

        report_text = classification_report(
            all_true,
            all_dt_pred_np,
            target_names=class_names,
            zero_division=0
        )

        with open(os.path.join(result_path, "classification_report_dt.txt"), "w") as f:
            f.write(report_text)

    report_df_nn = pd.DataFrame(
        classification_report(
            all_true,
            all_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    report_df_nn.to_csv(os.path.join(result_path, "classification_report_nn.csv"))

    report_text = classification_report(
        all_true,
        all_pred,
        target_names=class_names,
        zero_division=0
    )

    with open(os.path.join(result_path, "classification_report_nn.txt"), "w") as f:
        f.write(report_text)

    report_df_hyb = pd.DataFrame(
        classification_report(
            all_true,
            hybrid_pred_np,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    report_df_hyb.to_csv(os.path.join(result_path, "classification_report_hybrid.csv"))

    report_text = classification_report(
        all_true,
        hybrid_pred_np,
        target_names=class_names,
        zero_division=0
    )

    with open(os.path.join(result_path, "classification_report_hybrid.txt"), "w") as f:
        f.write(report_text)

    # ------------------------------------------------------------------ #
    #  Fallback debug output                                              #
    # ------------------------------------------------------------------ #
    fallback_indices = [
        i for i, conf in enumerate(all_dominant_probs)
        if conf < adaptive_threshold
    ]

    for i in fallback_indices:
        if not secondary_model:
            continue

        nn_pred = all_pred[i]
        true = all_true[i]
        nn_conf = all_dominant_probs[i]

        dt_pred = all_dt_pred[i]
        dt_conf = all_dt_confidence[i]
        dt_probs = all_dt_prob_vectors[i]

        use_xgb, decision = hybrid_decision(
            nn_pred, nn_conf, dt_pred, dt_conf, adaptive_threshold
        )

        delta = dt_conf - nn_conf
        sign = "+" if delta >= 0 else ""

        # status: only meaningful when an override was (or would be) attempted
        if use_xgb:
            status = "\u2705 FIX" if dt_pred == true else "\u274c BREAK"
        else:
            status = None

        print(f"[{i}]")
        print(f"NN={nn_pred} ({nn_conf:.3f})")
        print(f"{secondary_type.upper()}={dt_pred} ({dt_conf:.3f})")
        print(f"\u0394={sign}{delta:.3f}")
        print(f"Decision={decision}")
        if status is not None:
            print(f"Status={status}")
        print(
            f"   {secondary_type.upper()} probs: {np.round(dt_probs, 4)}"
        )
        print()

        # collect images where the secondary model hurts or both agree wrongly
        if nn_pred == true and dt_pred != true:
            misclassified_secondary.append({
                "image": all_images[i],
                "img_path": all_img_paths[i],
                "true": true,
                "pred": dt_pred,
                "prob_pred": dt_conf,
                "coords": all_coords_list[i]
            })
        elif nn_pred == dt_pred and nn_pred != true:
            misclassified_secondary.append({
                "image": all_images[i],
                "img_path": all_img_paths[i],
                "true": true,
                "pred": dt_pred,
                "prob_pred": dt_conf,
                "coords": all_coords_list[i]
            })

    # ------------------------------------------------------------------ #
    #  Hybrid statistics                                                  #
    # ------------------------------------------------------------------ #
    fallback_count = sum(
        info["nn_conf"] < adaptive_threshold for info in secondary_info
    )

    override_deltas_arr = np.array(override_deltas) if override_deltas else np.array([])

    print("=== HYBRID ANALYSIS ===")
    print(f"Adaptive threshold:        {adaptive_threshold:.4f}")
    print(f"Fallback samples:          {fallback_count}")
    print(f"Override count:            {secondary_used}")
    print(f"Agreement rejections:      {reject_agreement}")
    print(f"Low confidence rejections: {reject_low_confidence}")
    print(f"Margin rejections:         {reject_small_margin}")
    if secondary_used > 0:
        print(f"Override precision:        {successful_overrides / secondary_used:.4f}")
    print(f"Successful overrides:      {successful_overrides}")
    print(f"Failed overrides:          {failed_overrides}")
    print(f"Net gain:                  {successful_overrides - failed_overrides}")
    if len(override_deltas_arr) > 0:
        print(f"Mean delta:                {np.mean(override_deltas_arr):.4f}")
        print(f"Median delta:              {np.median(override_deltas_arr):.4f}")
        print(f"Min delta:                 {np.min(override_deltas_arr):.4f}")
        print(f"Max delta:                 {np.max(override_deltas_arr):.4f}")

    return (
        misclassified_nn,
        all_coords,
        all_dominant_probs,
        all_eval_paths,
        all_pred,
        all_true,
        all_prob_vectors,
        test_accuracy,
        hybrid_accuracy,
        misclassified_secondary
    )

    
def perform_all_patches_corrections(all_coords, all_dominant_probs, all_eval_paths, all_pred, all_true):
    for coords, prob, img_path, pred, true in zip(all_coords, all_dominant_probs, all_eval_paths, all_pred, all_true):
        print(prob)

def build_heatmap(H, W, patches_list, sigma=2):
    """
    Build normalized heatmap from patch confidence scores.
    Handles overlapping regions properly.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2
    })
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

def plot_misclassified_with_heatmap(results_path, misclassified_list, sigma=2, alpha=0.3, postfix=None):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2
    })

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
        if postfix:
            combined_path = os.path.join(
            results_path, "secondary_"+f"combined_{base}.png"
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
        if postfix:
            combined_path = os.path.join(
            boxes_path, "secondary_"+f"boxes_{base}.png"
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
        if postfix:
            combined_path = os.path.join(
            heatmap_path, "secondary_"+f"heatmap_{base}.png"
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
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2
    })

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
        # ax_img.set_title(f"Misclassified Patches - {item["img_path"]}")

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

            # # Add label
            # ax_img.text(
            #     x1,
            #     y1 - 5,
            #     f"T:{item['true']} P:{item['pred']}",
            #     color='yellow',
            #     fontsize=8,
            #     bbox=dict(facecolor='black', alpha=0.6)
            # )

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
        

def create_heatmaps_per_image(results_path, all_eval_paths, coords, probs, sigma=0):
    """
    Creates one confidence heatmap per evaluated image.
    all_eval_paths, coords, probs must be aligned.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2
    })

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

        # -----------------------------
        # Plot
        # -----------------------------
        
        np.save(os.path.join(results_path, "heatmap.npy"), heatmap)
        
        
        masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
        # Optional smoothing
        if sigma > 0:
            masked_heatmap = gaussian_filter(heatmap, sigma=sigma)

        vmin = masked_heatmap.min()
        vmax = masked_heatmap.max()

        fig, ax = plt.subplots(figsize=(6, 6))

        im = ax.imshow(masked_heatmap,
                    cmap="inferno",
                    origin="upper",
                    vmin=vmin,
                    vmax=vmax)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence")

        # ax.set_title(f"Confidence Heatmap\n{os.path.basename(img_path)}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        path_to_save = os.path.join(results_path, f"confidence_{base}.png")

        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
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
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2
    })

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

        np.save(os.path.join(results_path, "error_map.npy"), error_map)
    
        error_map = gaussian_filter(error_map, sigma=3)

        fig, ax = plt.subplots(figsize=(6, 6))

        im = ax.imshow(error_map,
                    cmap="magma",
                    origin="upper")
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence")
        # plt.title(f"Error Density\n{os.path.basename(img_path)}")

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
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2
    })

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
        # plt.title(f"Class {target_class} Error Map\n{os.path.basename(img_path)}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        path_to_save = os.path.join(results_path, f"class_{target_class}_error_{base}.png")
        plt.savefig(path_to_save, dpi=300)
        plt.close()
def create_uncertainty_maps(results_path, all_eval_paths, coords, prob_vectors, misclassified_list):
    """
    prob_vectors: list of softmax vectors per patch (shape: [N, num_classes])
    misclassified_list: list of dicts with keys:
        img_path, coords, true, pred, prob_pred
    """

    from collections import defaultdict
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2  # OpenCV for color map
    from scipy.ndimage import gaussian_filter
    import seaborn as sns
    import matplotlib.patches as patches

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2
    })

    img_to_indices = defaultdict(list)
    for i, path in enumerate(all_eval_paths):
        img_to_indices[path].append(i)

    # group misclassified patches by image
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

        # smooth
        uncertainty_map = gaussian_filter(uncertainty_map, sigma=2)

        # -------------------------
        # 1️⃣ Pure uncertainty map
        # -------------------------
        sns.set_theme(style="white")

        fig, ax = plt.subplots(figsize=(6, 6))

        im = ax.imshow(
            uncertainty_map,
            cmap=sns.color_palette("mako", as_cmap=True),
            origin="upper"
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Prediction Entropy")

        ax.set_title("Uncertainty Map")

        plt.tight_layout()

        path_to_save = os.path.join(results_path, f"uncertainty_{base}.png")
        plt.savefig(path_to_save, dpi=600, bbox_inches="tight")
        plt.close()

        # -------------------------
        # 2️⃣ Overlay heatmap + grayscale + boxes
        # -------------------------

        img_gray = np.array(img.convert("L"))

        # normalize uncertainty map
        unc_norm = uncertainty_map - uncertainty_map.min()
        if unc_norm.max() > 0:
            unc_norm = unc_norm / unc_norm.max()

        # -------------------------
        # 2.1 Generate the heatmap using OpenCV
        # -------------------------
        # Convert the normalized uncertainty map to a color heatmap using OpenCV's applyColorMap
        heatmap = cv2.applyColorMap((unc_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # ✅ convert
        # -------------------------
        # 2.2 Alpha blending: Grayscale image + Heatmap
        # -------------------------
        img_gray_rgb = np.stack([img_gray] * 3, axis=-1)  # Convert grayscale to RGB
        img_gray_rgb = cv2.cvtColor(img_gray_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Perform alpha blending with normalized uncertainty
        alpha = unc_norm[..., None]  # Reshape to match dimensions (H, W, 1)

        # Blend grayscale image with heatmap based on uncertainty
        img_with_uncertainty = cv2.convertScaleAbs(
            img_gray_rgb * (1 - alpha) + heatmap * alpha
        )

        # Ensure the resulting image is within valid range [0, 255]
        img_with_uncertainty = np.clip(img_with_uncertainty, 0, 255).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(8, 8))

        # show overlay image
        ax.imshow(img_with_uncertainty, origin="upper")

        # draw misclassified boxes
        for item in mis_by_img.get(img_path, []):
            x1, y1, x2, y2 = map(int, item["coords"])
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                label="Misclassified Patch"
            )
            ax.add_patch(rect)

        # --- create legend entry manually (so duplicates don't appear) ---
        legend_patch = patches.Patch(
            edgecolor="red",
            facecolor="none",
            linewidth=2,
            label="Misclassified Patch"
        )

        ax.legend(
            handles=[legend_patch],
            loc="upper right",
            frameon=True
        )

        # --- add colorbar for uncertainty ---
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(vmin=unc_norm.min(), vmax=unc_norm.max())
        sm = cm.ScalarMappable(norm=norm, cmap="magma")
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Prediction Uncertainty (Entropy)")

        # ax.set_title("Uncertainty + Misclassified Patches")
        ax.axis("off")

        plt.tight_layout()

        overlay_path = os.path.join(results_path, f"uncertainty_overlay_{base}.png")
        plt.savefig(overlay_path, dpi=600, bbox_inches="tight")
        plt.close()