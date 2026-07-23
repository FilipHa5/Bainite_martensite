import os
import numpy as np
import matplotlib.pyplot as plt


def reliability_diagram(y_true, probs, results_path=None, n_bins=10, prefix="primary"):
    probs = np.asarray(probs)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == np.asarray(y_true)

    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.sum(mask) == 0:
            continue
        accs.append(accuracies[mask].mean())
        confs.append(confidences[mask].mean())

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--")
    plt.plot(confs, accs, "o-")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid()

    if results_path is not None:
        path = os.path.join(results_path, f"{prefix}_reliability_diagram.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def confidence_histogram(probs, results_path, n_bins=20, prefix="primary"):
    confidences = np.max(np.asarray(probs), axis=1)

    plt.figure(figsize=(5, 5))
    plt.hist(confidences, bins=n_bins)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")
    plt.grid()

    path = os.path.join(results_path, f"{prefix}_confidence_histogram.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


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
    path = os.path.join(results_path, f"{prefix}_accuracy_vs_threshold.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # Coverage vs threshold
    plt.figure(figsize=(5, 5))
    plt.plot(thresholds, coverage)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Coverage")
    plt.title("Coverage")
    plt.grid()
    path = os.path.join(results_path, f"{prefix}_coverage_vs_threshold.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

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
    path = os.path.join(results_path, f"{prefix}_risk_coverage_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def expected_calibration_error(y_true, probs, n_bins=15):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == y_true

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.sum(mask) == 0:
            continue
        accuracy = accuracies[mask].mean()
        confidence = confidences[mask].mean()
        ece += np.abs(accuracy - confidence) * np.sum(mask) / len(y_true)

    return ece
