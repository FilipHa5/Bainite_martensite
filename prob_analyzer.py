import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Directory containing .out files
# ==========================================================

LOG_DIR = Path("ttest2")

# ==========================================================
# Output directory for plots
# ==========================================================

OUT_DIR = LOG_DIR / "analysis_plots"
OUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# Regex pattern
# ==========================================================

pattern = re.compile(
    r"\[(\d+)\]\s*"
    r"NN=(\d)\s*\(([0-9.]+)\)\s*"
    r"XGB=(\d)\s*\(([0-9.]+)\)\s*"
    r"Δ=([+-]?[0-9.]+)\s*"
    r"Decision=(.*?)\n"
    r"(?:Status=(.*?)\n)?"
    r"\s*XGB probs:\s*\[\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\]",
    re.MULTILINE
)

records = []

# ==========================================================
# Read every .out file
# ==========================================================

files = sorted(LOG_DIR.glob("*.out"))

print(f"Found {len(files)} .out files\n")

for file in files:

    text = file.read_text(errors="ignore")

    for m in pattern.finditer(text):

        p0 = float(m.group(9))
        p1 = float(m.group(10))

        records.append({
            "file": file.name,
            "index": int(m.group(1)),
            "nn_pred": int(m.group(2)),
            "nn_conf": float(m.group(3)),
            "xgb_pred": int(m.group(4)),
            "xgb_conf": float(m.group(5)),
            "delta": float(m.group(6)),
            "decision": m.group(7).strip(),
            "status": "" if m.group(8) is None else m.group(8).strip(),
            "prob0": p0,
            "prob1": p1,
            "dominant_prob": max(p0, p1)
        })

df = pd.DataFrame(records)

print("=" * 70)
print(f"Parsed {len(df):,} predictions from {len(files)} files")
print("=" * 70)

# ==========================================================
# Overall statistics
# ==========================================================

dom = df["dominant_prob"]

print("\nOverall probability statistics")
print(dom.describe())

print("\nPercentiles")

for p in [1,5,10,20,25,50,75,80,90,95,99]:
    print(f"{p:2d}% : {np.percentile(dom,p):.4f}")

adaptive = np.clip(np.percentile(dom,10),0.75,0.90)

print("\nAdaptive threshold")
print("------------------")
print("10th percentile :", np.percentile(dom,10))
print("Threshold       :", adaptive)

# ==========================================================
# Decision statistics
# ==========================================================

print("\nDecision counts")
print(df["decision"].value_counts())

print("\nDecision statistics")

stats = (
    df.groupby("decision")["dominant_prob"]
      .agg(["count","mean","median","std","min","max"])
      .sort_values("count", ascending=False)
)

print(stats)

# ==========================================================
# Per-file summary
# ==========================================================

print("\nPer-file summary")

summary = (
    df.groupby("file")
      .agg(
          samples=("file","size"),
          mean_prob=("dominant_prob","mean"),
          median_prob=("dominant_prob","median"),
          threshold=("dominant_prob",
                     lambda x: np.clip(np.percentile(x,10),0.75,0.90))
      )
)

print(summary)

# ==========================================================
# Lowest confidence samples
# ==========================================================

print("\nLowest confidence predictions")

print(
    df.sort_values("dominant_prob")
      .head(30)[[
          "file",
          "index",
          "decision",
          "dominant_prob",
          "nn_conf",
          "xgb_conf",
          "delta"
      ]]
)

# ==========================================================
# Histogram
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(dom, bins=30, edgecolor="black")

plt.axvline(adaptive,
            color="red",
            linestyle="--",
            label=f"Adaptive={adaptive:.3f}")

plt.xlabel("Dominant probability")
plt.ylabel("Count")
plt.title("Distribution of dominant XGB probabilities")
plt.legend()

plt.tight_layout()
plt.savefig(
    OUT_DIR / "dominant_probability_histogram.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ==========================================================
# Confidence by decision
# ==========================================================

plt.figure(figsize=(12,5))

df.boxplot(column="dominant_prob", by="decision", grid=False)

plt.ylabel("Dominant probability")
plt.title("Confidence by decision")
plt.suptitle("")

plt.tight_layout()
plt.savefig(
    OUT_DIR / "confidence_by_decision_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ==========================================================
# Scatter
# ==========================================================

plt.figure(figsize=(6,6))

plt.scatter(df["nn_conf"], df["xgb_conf"], alpha=0.5)

plt.xlabel("NN confidence")
plt.ylabel("XGB confidence")
plt.title("NN vs XGB confidence")

plt.plot([0.5,1],[0.5,1],'r--')

plt.tight_layout()
plt.savefig(
    OUT_DIR / "nn_vs_xgb_confidence.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
