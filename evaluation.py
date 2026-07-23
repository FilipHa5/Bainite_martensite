import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, classification_report

from calibration import (
    accuracy_vs_threshold,
    confidence_histogram,
    expected_calibration_error,
    reliability_diagram,
    risk_coverage_curve,
)
from models import normalized_histogram


def _secondary_forward(secondary_model, secondary_type, rgb_tensor, lbp_tensor, n_bins):
    if secondary_model is None:
        return {"pred": None, "conf": None, "probs": None}

    if secondary_type == "xgb":
        rgb_hist = normalized_histogram(rgb_tensor.cpu(), bins=n_bins)
        lbp_hist = (
            normalized_histogram(lbp_tensor.cpu(), bins=n_bins)
            if lbp_tensor is not None
            else np.zeros(n_bins, dtype=np.float32)
        )
        features = np.concatenate([rgb_hist, lbp_hist]).reshape(1, -1)
        pred = secondary_model.predict(features)[0]
        probs = secondary_model.predict_proba(features)[0]
        return {"pred": pred, "conf": probs[pred], "probs": probs}

    if secondary_type == "cnn":
        if lbp_tensor is not None:
            logits = secondary_model(rgb_tensor.unsqueeze(0), lbp_tensor.unsqueeze(0))
        else:
            logits = secondary_model(rgb_tensor.unsqueeze(0))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        return {"pred": pred, "conf": float(probs[pred]), "probs": probs}

    raise ValueError(f"Unknown secondary_type: {secondary_type}")


def _hybrid_decision(nn_pred, nn_conf, dt_pred, dt_conf,
                      adaptive_threshold, confidence_margin, min_secondary_confidence):
    if dt_pred is None:
        return False, "No secondary model"
    if nn_conf >= adaptive_threshold:
        return False, "High NN confidence"
    if nn_pred == dt_pred:
        return False, "Agreement"
    if min_secondary_confidence is not None and dt_conf < min_secondary_confidence:
        return False, "Low secondary confidence"
    if (dt_conf - nn_conf) < confidence_margin:
        return False, "Small confidence margin"
    return True, "Override"


def _save_report(y_true, y_pred, class_names, result_path, tag):
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(result_path, f"classification_report_{tag}.csv")
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    with open(os.path.join(result_path, f"classification_report_{tag}.txt"), "w") as f:
        f.write(report_text)


@torch.no_grad()
def compute_hybrid_predictions(model, loader, device, secondary_model=None, secondary_type="xgb", n_bins=16):
    model.eval()
    if secondary_model and secondary_type == "cnn":
        secondary_model.eval()

    all_true, all_pred, all_dominant_probs = [], [], []
    all_dt_pred, all_dt_confidence = [], []

    for batch in loader:
        rgb = batch["rgb"].to(device)
        labels = batch["label"].to(device)
        lbp = batch["lbp"].to(device) if batch.get("lbp") is not None else None

        logits = model(rgb, lbp) if lbp is not None else model(rgb)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        for i in range(rgb.size(0)):
            all_true.append(labels[i].item())
            all_pred.append(preds[i].item())
            all_dominant_probs.append(probs[i][preds[i]].item())

            sec_out = _secondary_forward(
                secondary_model, secondary_type,
                rgb[i], lbp[i] if lbp is not None else None, n_bins
            )
            all_dt_pred.append(sec_out["pred"])
            all_dt_confidence.append(sec_out["conf"])

    return {
        "all_true": all_true,
        "all_pred": all_pred,
        "all_dominant_probs": all_dominant_probs,
        "all_dt_pred": all_dt_pred,
        "all_dt_confidence": all_dt_confidence,
    }


def compute_hybrid_accuracy(predictions, confidence_margin, min_conf_threshold, min_secondary_confidence=0.75):
    all_true = np.array(predictions["all_true"])
    all_pred = np.array(predictions["all_pred"])
    all_dominant_probs = predictions["all_dominant_probs"]
    all_dt_pred = predictions["all_dt_pred"]
    all_dt_confidence = predictions["all_dt_confidence"]

    # adaptive_threshold = np.clip(
    #     np.percentile(all_dominant_probs, min_conf_threshold), 0.75, 0.90
    # )
    adaptive_threshold = min_conf_threshold

    hybrid_pred = []
    for i in range(len(all_true)):
        use_secondary, _ = _hybrid_decision(
            all_pred[i], all_dominant_probs[i],
            all_dt_pred[i], all_dt_confidence[i],
            adaptive_threshold, confidence_margin, min_secondary_confidence,
        )
        hybrid_pred.append(all_dt_pred[i] if use_secondary else all_pred[i])

    return (np.array(hybrid_pred) == all_true).mean()


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
    min_conf_threshold=10,
):
    model.eval()
    if secondary_model and secondary_type == "cnn":
        secondary_model.eval()

    # --- Collect predictions from all batches ---
    all_true, all_pred, all_dominant_probs, all_prob_vectors = [], [], [], []
    all_coords, all_eval_paths, all_margins = [], [], []
    all_images, all_img_paths, all_coords_list = [], [], []
    all_dt_pred, all_dt_confidence, all_dt_prob_vectors = [], [], []
    misclassified_nn, secondary_info = [], []

    for batch in loader:
        rgb = batch["rgb"].to(device)
        labels = batch["label"].to(device)
        coords = batch["coords"].to(device)
        img_paths = batch["img_path"]
        lbp = batch["lbp"].to(device) if batch.get("lbp") is not None else None

        logits = model(rgb, lbp) if lbp is not None else model(rgb)
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

            sec_out = _secondary_forward(
                secondary_model, secondary_type,
                rgb[i], lbp[i] if lbp is not None else None, n_bins
            )
            dt_pred, dt_conf, dt_probs = sec_out["pred"], sec_out["conf"], sec_out["probs"]

            if secondary_model:
                all_dt_pred.append(dt_pred)
                all_dt_confidence.append(dt_conf)
                all_dt_prob_vectors.append(dt_probs)

            if nn_pred != true_label and len(misclassified_nn) < max_show:
                misclassified_nn.append({
                    "image": rgb[i].cpu(), "img_path": img_paths[i],
                    "true": true_label, "pred": nn_pred,
                    "prob_pred": confidence, "coords": coords[i].cpu().tolist(),
                })

            secondary_info.append({
                "nn_pred": nn_pred, "nn_conf": confidence,
                "dt_pred": dt_pred, "dt_conf": dt_conf, "true": true_label,
            })

    # --- Calibration metrics ---
    brier_nn = brier_score_loss(np.array(all_true), np.array(all_prob_vectors)[:, 1])
    print(f"Brier score NN: {brier_nn}")
    
    brier_secondary = brier_score_loss(np.array(all_true), np.array(all_dt_prob_vectors)[:, 1])
    print(f"Brier score secondary: {brier_secondary}")

    ece_nn = expected_calibration_error(all_true, np.array(all_prob_vectors), n_bins=15)
    print(f"ECE NN: {ece_nn}")

    ece_secondary = expected_calibration_error(all_true, np.array(all_dt_prob_vectors), n_bins=15)
    print(f"ECE NN: {ece_secondary}")

    # primary
    reliability_diagram(all_true, np.array(all_prob_vectors), result_path)
    confidence_histogram(np.array(all_prob_vectors), result_path)
    thresholds, accuracy_arr, coverage_arr = accuracy_vs_threshold(
        all_true, np.array(all_prob_vectors), result_path
    )
    risk_coverage_curve(accuracy_arr, coverage_arr, result_path)

    # secondary
    reliability_diagram(all_true, np.array(all_dt_prob_vectors), result_path, prefix="secondary")
    confidence_histogram(np.array(all_dt_prob_vectors), result_path, prefix="secondary")
    thresholds, accuracy_arr, coverage_arr = accuracy_vs_threshold(
        all_true, np.array(all_dt_prob_vectors), result_path, prefix="secondary"
    )
    risk_coverage_curve(accuracy_arr, coverage_arr, result_path, prefix="secondary")

    # --- Hybrid gating ---
    # adaptive_threshold = np.clip(
    #     np.percentile(all_dominant_probs, min_conf_threshold), 0.75, 0.90
    # )
    adaptive_threshold = min_conf_threshold

    hybrid_pred = []
    successful_overrides = 0
    failed_overrides = 0
    secondary_used = 0
    reject_agreement = 0
    reject_low_confidence = 0
    reject_small_margin = 0
    override_deltas = []

    for info in secondary_info:
        use_secondary, decision = _hybrid_decision(
            info["nn_pred"], info["nn_conf"],
            info["dt_pred"], info["dt_conf"],
            adaptive_threshold, confidence_margin, min_secondary_confidence,
        )

        if use_secondary:
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
                elif decision == "Low secondary confidence":
                    reject_low_confidence += 1
                elif decision == "Small confidence margin":
                    reject_small_margin += 1

    # --- Accuracy ---
    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)
    hybrid_pred_np = np.array(hybrid_pred)

    test_accuracy = (all_pred_np == all_true_np).mean()
    hybrid_accuracy = (hybrid_pred_np == all_true_np).mean()

    # --- Classification reports ---
    _save_report(all_true, all_pred, class_names, result_path, "nn")
    _save_report(all_true, hybrid_pred_np, class_names, result_path, "hybrid")
    if secondary_model and len(all_dt_pred) > 0:
        _save_report(all_true, np.array(all_dt_pred), class_names, result_path, "dt")

    # --- Fallback debug output ---
    misclassified_secondary = []
    fallback_indices = [
        i for i, conf in enumerate(all_dominant_probs) if conf < adaptive_threshold
    ]

    for i in fallback_indices:
        if not secondary_model:
            continue

        nn_pred, true, nn_conf = all_pred[i], all_true[i], all_dominant_probs[i]
        dt_pred, dt_conf, dt_probs = all_dt_pred[i], all_dt_confidence[i], all_dt_prob_vectors[i]

        use_secondary, decision = _hybrid_decision(
            nn_pred, nn_conf, dt_pred, dt_conf,
            adaptive_threshold, confidence_margin, min_secondary_confidence,
        )

        delta = dt_conf - nn_conf
        sign = "+" if delta >= 0 else ""

        if use_secondary:
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
        print(f"   {secondary_type.upper()} probs: {np.round(dt_probs, 4)}")
        print()

        if (nn_pred == true and dt_pred != true) or (nn_pred == dt_pred and nn_pred != true):
            misclassified_secondary.append({
                "image": all_images[i], "img_path": all_img_paths[i],
                "true": true, "pred": dt_pred,
                "prob_pred": dt_conf, "coords": all_coords_list[i],
            })

    # --- Hybrid statistics ---
    fallback_count = sum(info["nn_conf"] < adaptive_threshold for info in secondary_info)
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
        misclassified_secondary,
    )
