import re
import os
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 🎨 GLOBAL STYLE (Publication)
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

# Muted scientific palette
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

    data = defaultdict(lambda: defaultdict(list))

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

                # 🚨 Start a new run explicitly when epoch == 1
                if epoch == 1:
                    current_run = {
                        "epochs": [],
                        "train_loss": [],
                        "val_loss": [],
                        "train_acc": [],
                        "val_acc": [],
                    }
                    data[current_outer][current_params].append(current_run)

                # Now append safely
                current_run["epochs"].append(epoch)
                current_run["train_loss"].append(train_loss)
                current_run["val_loss"].append(val_loss)
                current_run["train_acc"].append(train_acc)
                current_run["val_acc"].append(val_acc)

            param_match = param_pattern.search(line)
            if param_match:
                lr = float(param_match.group(1))
                wd = float(param_match.group(2))
                current_params = (lr, wd)

                # IMPORTANT: force new run block
                current_run = None

                continue

    return data


# -----------------------------
# PLOTTING FUNCTION
# -----------------------------

def plot_learning_curves(data, output_dir="learning_curves"):

    os.makedirs(output_dir, exist_ok=True)

    for outer_fold, param_dict in data.items():
        for (lr, wd), runs in param_dict.items():

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Colors (muted scientific style)
            train_color = palette[0]  # navy-ish
            val_color = palette[3]    # muted red-ish

            # ---------------- LOSS ----------------
            ax = axes[0]

            for run in runs:
                ax.plot(
                    run["epochs"],
                    run["train_loss"],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    color=train_color,
                )
                ax.plot(
                    run["epochs"],
                    run["val_loss"],
                    linestyle="-",
                    linewidth=1.8,
                    alpha=0.9,
                    color=val_color,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Loss\nFold {outer_fold} | LR={lr:.0e}, WD={wd:.0e}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # ---------------- ACCURACY ----------------
            ax = axes[1]

            for run in runs:
                ax.plot(
                    run["epochs"],
                    run["train_acc"],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    color=train_color,
                )
                ax.plot(
                    run["epochs"],
                    run["val_acc"],
                    linestyle="-",
                    linewidth=1.8,
                    alpha=0.9,
                    color=val_color,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Accuracy\nFold {outer_fold} | LR={lr:.0e}, WD={wd:.0e}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Unified legend (clean)
            lines = [
                plt.Line2D([0], [0], color=train_color, linestyle="--", lw=1.5),
                plt.Line2D([0], [0], color=val_color, linestyle="-", lw=1.8),
            ]
            labels = ["Train", "Validation"]
            fig.legend(lines, labels, loc="lower center", ncol=2, frameon=False)

            plt.tight_layout(rect=[0, 0.08, 1, 1])

            filename = f"outer{outer_fold}_lr{lr:.0e}_wd{wd:.0e}.svg"
            filepath = os.path.join(output_dir, filename)

            plt.savefig(filepath, bbox_inches="tight")
            plt.close()

            print(f"Saved: {filepath}")


# -----------------------------
# MAIN
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=str)
    parser.add_argument("--out", type=str, default="learning_curves")
    args = parser.parse_args()

    data = parse_log_file(args.logfile)
    plot_learning_curves(data, args.out)


if __name__ == "__main__":
    main()