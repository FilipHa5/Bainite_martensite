import os
import json
import numpy as np
import matplotlib.pyplot as plt


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return [_to_serializable(x) for x in value.tolist()]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, list):
        return [_to_serializable(x) for x in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


def _save_plot(results_path, prefix, filename):
    if results_path is None:
        return
    os.makedirs(results_path, exist_ok=True)
    path = os.path.join(results_path, f"{prefix}_{filename}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_json(results_path, filename, data):
    if results_path is None:
        return
    os.makedirs(results_path, exist_ok=True)
    path = os.path.join(results_path, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=4)


def reliability_diagram(y_true, probs, results_path=None, n_bins=10, prefix="primary"):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == y_true

    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.sum(mask) == 0:
            continue
        accs.append(accuracies[mask].mean())
        confs.append(confidences[mask].mean())

    confs = np.asarray(confs)
    accs = np.asarray(accs)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--")
    plt.plot(confs, accs, "o-")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid()

    _save_plot(results_path, prefix, "reliability_diagram")
    _save_json(results_path, f"{prefix}_reliability_diagram.json", {
        "confidence": confs,
        "accuracy": accs,
        "n_bins": n_bins,
    })

    return confs, accs


def confidence_histogram(probs, results_path, n_bins=20, prefix="primary"):
    confidences = np.max(np.asarray(probs), axis=1)

    plt.figure(figsize=(5, 5))
    counts, edges, _ = plt.hist(confidences, bins=n_bins)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")
    plt.grid()

    _save_plot(results_path, prefix, "confidence_histogram")
    _save_json(results_path, f"{prefix}_confidence_histogram.json", {
        "counts": counts,
        "bin_edges": edges,
        "n_bins": n_bins,
    })

    return counts, edges


def accuracy_vs_threshold(y_true, probs, results_path, n_thresholds=50, prefix="primary"):
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    true = np.asarray(y_true)

    thresholds = np.linspace(0, 1, n_thresholds)
    accuracy, coverage = [], []

    for t in thresholds:
        mask = conf >= t
        if mask.sum() == 0:
            accuracy.append(np.nan)
            coverage.append(0)
            continue
        accuracy.append((pred[mask] == true[mask]).mean())
        coverage.append(mask.mean())

    thresholds = np.asarray(thresholds)
    accuracy = np.asarray(accuracy)
    coverage = np.asarray(coverage)

    # Accuracy vs threshold
    plt.figure(figsize=(5, 5))
    plt.plot(thresholds, accuracy)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Accuracy")
    plt.title("Selective Accuracy")
    plt.grid()
    _save_plot(results_path, prefix, "accuracy_vs_threshold")

    # Coverage vs threshold
    plt.figure(figsize=(5, 5))
    plt.plot(thresholds, coverage)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Coverage")
    plt.title("Coverage")
    plt.grid()
    _save_plot(results_path, prefix, "coverage_vs_threshold")

    _save_json(results_path, f"{prefix}_accuracy_vs_threshold.json", {
        "threshold": thresholds,
        "accuracy": accuracy,
        "coverage": coverage,
    })

    return thresholds, accuracy, coverage


def risk_coverage_curve(accuracy, coverage, results_path, prefix="primary"):
    accuracy = np.asarray(accuracy)
    coverage = np.asarray(coverage)
    risk = 1 - accuracy

    # Drop NaN entries (where coverage was 0)
    valid = ~np.isnan(risk)
    plt.figure(figsize=(5, 5))
    plt.plot(coverage[valid], risk[valid])
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title("Risk-Coverage Curve")
    plt.grid()

    _save_plot(results_path, prefix, "risk_coverage_curve")
    _save_json(results_path, f"{prefix}_risk_coverage_curve.json", {
        "coverage": coverage[valid],
        "risk": risk[valid],
    })

    return risk[valid], coverage[valid]


def expected_calibration_error(y_true, probs, n_bins=15, results_path=None, prefix="primary"):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == y_true

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    bin_confidences, bin_accuracies, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.sum(mask) == 0:
            continue
        accuracy = accuracies[mask].mean()
        confidence = confidences[mask].mean()
        bin_counts.append(int(np.sum(mask)))
        bin_confidences.append(confidence)
        bin_accuracies.append(accuracy)
        ece += np.abs(accuracy - confidence) * np.sum(mask) / len(y_true)

    _save_json(results_path, f"{prefix}_ece.json", {
        "ece": ece,
        "n_bins": n_bins,
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    })

    return ece
