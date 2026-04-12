import re
import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import calibration_curve

# -----------------------------
# GLOBAL STYLE
# -----------------------------

sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

palette = sns.color_palette("deep")


# -----------------------------
# LOG PARSER
# -----------------------------

def parse_log_file(filepath):

    outer_fold_pattern = re.compile(r"=+ OUTER FOLD (\d+) =+")
    param_pattern = re.compile(r"Inner loop NN, LR: ([\d.eE-]+), WD: ([\d.eE-]+)")
    epoch_pattern = re.compile(
        r"Epoch (\d+)/\d+ \| "
        r"Train Loss: ([\d.]+), Train Acc: ([\d.]+)% \| "
        r"Val Loss: ([\d.]+), Val Acc: ([\d.]+)%"
    )

    confidence_pattern = re.compile(
        r"Confidence: ([\d.]+);XGBClassifier (worked well|Fucked up)\. Pred: (\d), true: (\d)"
    )
    
    fallback_pattern = re.compile(
        r"\[(\d+)\]\s+NN:\s+pred=(\d+)\s+conf=([\d.]+)\s+\|\s+XGB:\s+pred=(\d+)\s+conf=([\d.]+)\s+\|\s+TRUE=(\d+)"
    )

    nn_score_pattern = re.compile(r"Outer test score NN: ([\d.]+)")
    dt_score_pattern = re.compile(r"Outer test score DT: ([\d.]+)")
    hybrid_pattern = re.compile(r"Hybrid classification accuracy: ([\d.]+)")

    data = defaultdict(lambda: defaultdict(list))
    xgb_results = defaultdict(list)
    fallback_results = defaultdict(list)
    summary = defaultdict(dict)

    current_outer = None
    current_params = None
    current_run = None

    with open(filepath, "r") as f:

        for line in f:
            line = line.strip()

            outer_match = outer_fold_pattern.search(line)
            if outer_match:
                current_outer = int(outer_match.group(1))
                continue

            param_match = param_pattern.search(line)
            if param_match:
                lr = float(param_match.group(1))
                wd = float(param_match.group(2))

                current_params = (lr, wd)
                current_run = None
                continue

            epoch_match = epoch_pattern.search(line)
            if epoch_match:

                epoch = int(epoch_match.group(1))
                train_loss = float(epoch_match.group(2))
                train_acc = float(epoch_match.group(3))
                val_loss = float(epoch_match.group(4))
                val_acc = float(epoch_match.group(5))

                if epoch == 1:

                    current_run = {
                        "epochs": [],
                        "train_loss": [],
                        "val_loss": [],
                        "train_acc": [],
                        "val_acc": []
                    }

                    data[current_outer][current_params].append(current_run)

                current_run["epochs"].append(epoch)
                current_run["train_loss"].append(train_loss)
                current_run["val_loss"].append(val_loss)
                current_run["train_acc"].append(train_acc)
                current_run["val_acc"].append(val_acc)

            conf_match = confidence_pattern.search(line)
            if conf_match and current_outer is not None:

                conf = float(conf_match.group(1))
                pred = int(conf_match.group(3))
                true = int(conf_match.group(4))

                xgb_results[current_outer].append({
                    "confidence": conf,
                    "pred": pred,
                    "true": true,
                    "correct": pred == true
                })
            fb_match = fallback_pattern.search(line)
            if fb_match and current_outer is not None:

                patch_id = int(fb_match.group(1))
                nn_pred = int(fb_match.group(2))
                nn_conf = float(fb_match.group(3))
                xgb_pred = int(fb_match.group(4))
                xgb_conf = float(fb_match.group(5))
                true = int(fb_match.group(6))

                fallback_results[current_outer].append({
                    "patch": patch_id,
                    "nn_conf": nn_conf,
                    "xgb_conf": xgb_conf,
                    "nn_correct": nn_pred == true,
                    "xgb_correct": xgb_pred == true,
                })
            nn_match = nn_score_pattern.search(line)
            if nn_match:
                summary[current_outer]["nn_score"] = float(nn_match.group(1))

            dt_match = dt_score_pattern.search(line)
            if dt_match:
                summary[current_outer]["dt_score"] = float(dt_match.group(1))

            hybrid_match = hybrid_pattern.search(line)
            if hybrid_match:
                summary[current_outer]["hybrid_score"] = float(hybrid_match.group(1))

    return data, xgb_results, fallback_results, summary 

def plot_nn_vs_xgb(df, outdir):

    if df.empty:
        return

    plt.figure(figsize=(5,5))

    sns.scatterplot(
        data=df,
        x="nn_conf",
        y="xgb_conf",
        hue="xgb_correct",
        palette=[palette[3], palette[0]],
        alpha=0.7
    )

    plt.plot([0,1],[0,1],"--", color="gray")

    plt.xlabel("NN Confidence")
    plt.ylabel("XGB Confidence")
    plt.title("Fallback: NN vs XGB")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "nn_vs_xgb_scatter.svg"))
    plt.close()

def plot_confidence_gap(df, outdir):

    if df.empty:
        return

    df = df.copy()
    df["gap"] = df["xgb_conf"] - df["nn_conf"]

    plt.figure(figsize=(6,4))

    sns.histplot(df["gap"], bins=30, kde=True)
    plt.axvline(0, linestyle="--", color="red")

    plt.xlabel("XGB - NN Confidence")
    plt.title("Confidence Gap (Fallback)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confidence_gap.svg"))
    plt.close()
    
def plot_fallback_outcomes(df, outdir):

    if df.empty:
        return

    cases = []

    for _, r in df.iterrows():
        if not r["nn_correct"] and r["xgb_correct"]:
            cases.append("XGB fixes NN")
        elif r["nn_correct"] and not r["xgb_correct"]:
            cases.append("XGB breaks NN")
        else:
            cases.append("Same")

    df = df.copy()
    df["case"] = cases

    plt.figure(figsize=(6,4))

    sns.countplot(data=df, x="case")

    plt.title("Fallback Outcomes")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fallback_outcomes.svg"))
    plt.close()
    
def build_fallback_dataframe(fallback_results):

    rows = []

    for fold, results in fallback_results.items():
        for r in results:
            rows.append({
                "fold": fold,
                "nn_conf": r["nn_conf"],
                "xgb_conf": r["xgb_conf"],
                "nn_correct": "Correct" if r["nn_correct"] else "Wrong",
                "xgb_correct": "Correct" if r["xgb_correct"] else "Wrong",
            })

    return pd.DataFrame(rows)

# -----------------------------
# LEARNING CURVES
# -----------------------------

def plot_learning_curves(data, outdir):

    os.makedirs(outdir, exist_ok=True)

    for outer_fold, param_dict in data.items():

        for (lr, wd), runs in param_dict.items():

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            train_color = palette[0]
            val_color = palette[3]

            # LOSS
            ax = axes[0]

            for run in runs:

                ax.plot(run["epochs"], run["train_loss"],
                        linestyle="--", color=train_color)

                ax.plot(run["epochs"], run["val_loss"],
                        linestyle="-", color=val_color)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Loss\nFold {outer_fold} | LR={lr:.0e}, WD={wd:.0e}")

            # ACCURACY
            ax = axes[1]

            for run in runs:

                ax.plot(run["epochs"], run["train_acc"],
                        linestyle="--", color=train_color)

                ax.plot(run["epochs"], run["val_acc"],
                        linestyle="-", color=val_color)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Accuracy\nFold {outer_fold}")

            lines = [
                plt.Line2D([0], [0], color=train_color, linestyle="--"),
                plt.Line2D([0], [0], color=val_color, linestyle="-")
            ]

            fig.legend(lines, ["Train", "Validation"],
                       loc="lower center", ncol=2, frameon=False)

            plt.tight_layout(rect=[0, 0.08, 1, 1])

            fname = f"outer{outer_fold}_lr{lr:.0e}_wd{wd:.0e}.svg"
            plt.savefig(os.path.join(outdir, fname))
            plt.close()


# -----------------------------
# XGB DATAFRAME
# -----------------------------

def build_dataframe(xgb_results):

    rows = []

    for fold, results in xgb_results.items():

        for r in results:

            rows.append({
                "fold": fold,
                "confidence": r["confidence"],
                "correct": "Correct" if r["correct"] else "Wrong"
            })

    return pd.DataFrame(rows)


# -----------------------------
# CONFIDENCE DISTRIBUTION
# -----------------------------
def plot_confidence_distribution(df, outdir, conf_col="nn_conf", correct_col="nn_correct"):
    """
    Plot histogram of confidences for either NN or secondary model.
    """
    plt.figure(figsize=(6,4))

    sns.histplot(
        data=df,
        x=conf_col,
        hue=correct_col,
        bins=20,
        kde=True,
        palette=[palette[0], palette[3]]
    )

    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"{conf_col.upper()} Confidence Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{conf_col}_confidence_distribution.svg"))
    plt.close()

# -----------------------------
# CALIBRATION CURVE
# -----------------------------

def plot_calibration(xgb_results, outdir):

    conf = []
    correct = []

    for fold in xgb_results:

        for r in xgb_results[fold]:

            conf.append(r["confidence"])
            correct.append(r["correct"])

    prob_true, prob_pred = calibration_curve(correct, conf, n_bins=10)

    plt.figure(figsize=(5,5))

    plt.plot(prob_pred, prob_true, marker="o", label="XGB")
    plt.plot([0,1],[0,1],"--", color="gray", label="Perfect")

    plt.xlabel("Predicted Confidence")
    plt.ylabel("True Accuracy")
    plt.title("Calibration Curve")

    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, "calibration_curve.svg"))
    plt.close()


# -----------------------------
# CONFIDENCE SCATTER
# -----------------------------

def plot_confidence_scatter(xgb_results, outdir):

    xs = []
    ys = []

    for fold in xgb_results:

        for r in xgb_results[fold]:

            xs.append(r["confidence"])
            ys.append(1 if r["correct"] else 0)

    plt.figure(figsize=(6,4))

    plt.scatter(xs, ys, alpha=0.6)

    plt.xlabel("Confidence")
    plt.ylabel("Correct (1) / Wrong (0)")
    plt.title("Confidence vs Correctness")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confidence_scatter.svg"))
    plt.close()


# -----------------------------
# CONFIDENCE THRESHOLD CURVE
# -----------------------------

def plot_conf_threshold(xgb_results, outdir):

    conf = []
    correct = []

    for fold in xgb_results:

        for r in xgb_results[fold]:

            conf.append(r["confidence"])
            correct.append(r["correct"])

    thresholds = np.linspace(0.5,1,20)

    accs = []

    for t in thresholds:

        mask = [c >= t for c in conf]

        if sum(mask) == 0:
            accs.append(np.nan)
            continue

        vals = [correct[i] for i in range(len(correct)) if mask[i]]
        accs.append(np.mean(vals))

    plt.figure(figsize=(6,4))

    plt.plot(thresholds, accs, marker="o")

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Confidence Threshold")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confidence_threshold_curve.svg"))
    plt.close()


# -----------------------------
# BOX PLOT PER FOLD
# -----------------------------

def plot_fold_box(df, outdir):

    plt.figure(figsize=(6,4))

    sns.boxplot(
        data=df,
        x="fold",
        y="confidence",
        hue="correct"
    )

    plt.title("Confidence Distribution per Fold")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confidence_boxplot.svg"))
    plt.close()


# -----------------------------
# SUMMARY BARPLOT
# -----------------------------

def plot_model_comparison(summary, outdir):

    rows = []

    for fold in summary:

        rows.append({
            "fold": fold,
            "model": "NN",
            "score": summary[fold].get("nn_score", None)
        })

        rows.append({
            "fold": fold,
            "model": "DT",
            "score": summary[fold].get("dt_score", None)
        })

        rows.append({
            "fold": fold,
            "model": "Hybrid",
            "score": summary[fold].get("hybrid_score", None)
        })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(6,4))

    sns.barplot(data=df, x="fold", y="score", hue="model")

    plt.ylabel("Accuracy")
    plt.title("Model Comparison per Fold")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "model_comparison.svg"))
    plt.close()

# -----------------------------
# MAIN
# -----------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("--out", default="analysis")

    args = parser.parse_args()

    curves_dir = os.path.join(args.out, "learning_curves")
    xgb_dir = os.path.join(args.out, "xgb_analysis")

    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(xgb_dir, exist_ok=True)

    # -----------------------------
    # Parse logs
    # -----------------------------
    data, xgb_results, fallback_results, summary = parse_log_file(args.logfile)

    # -----------------------------
    # Build dataframes
    # -----------------------------
    df_xgb = build_dataframe(xgb_results)                     # For XGB-only data
    df_fb = build_fallback_dataframe(fallback_results)       # For fallback data (NN vs XGB)

    # -----------------------------
    # Plot learning curves
    # -----------------------------
    plot_learning_curves(data, curves_dir)

    # -----------------------------
    # XGB-specific plots
    # -----------------------------
    if not df_xgb.empty:
        plot_confidence_distribution(df_xgb, xgb_dir)           # XGB-only
        plot_calibration(xgb_results, xgb_dir)
        plot_confidence_scatter(xgb_results, xgb_dir)
        plot_conf_threshold(xgb_results, xgb_dir)
        plot_fold_box(df_xgb, xgb_dir)

    # -----------------------------
    # Fallback/hybrid plots
    # -----------------------------
    if not df_fb.empty:
        # Confidence distributions
        plot_confidence_distribution(df_fb, xgb_dir, conf_col="nn_conf", correct_col="nn_correct")
        plot_confidence_distribution(df_fb, xgb_dir, conf_col="xgb_conf", correct_col="xgb_correct")

        # Boxplots
        plot_fold_box(df_fb, xgb_dir)  # modify function if needed to handle nn_conf/xgb_conf columns

        # Custom fallback/hybrid plots
        plot_nn_vs_xgb(df_fb, xgb_dir)
        plot_confidence_gap(df_fb, xgb_dir)
        plot_fallback_outcomes(df_fb, xgb_dir)

    # -----------------------------
    # Model comparison summary
    # -----------------------------
    plot_model_comparison(summary, xgb_dir)

    print("Analysis finished.")


if __name__ == "__main__":
    main()