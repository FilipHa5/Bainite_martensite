import re
import pandas as pd
import numpy as np

def parse_log_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    data = []
    current_probs = None

    prob_pattern = re.compile(r"XGB probs:\s*\[([0-9.eE+-]+)\s+([0-9.eE+-]+)\]")
    row_pattern = re.compile(
        r"\[(\d+)\]\s+NN:\s+(\d)\s+\(([\d.]+)\)\s+\|\s+XGB:\s+(\d)\s+\(([\d.]+)\)\s+\|\s+TRUE=(\d)\s+\|\s+(.+)"
    )

    for line in lines:
        line = line.strip()

        # Match probabilities line
        prob_match = prob_pattern.search(line)
        if prob_match:
            current_probs = [float(prob_match.group(1)), float(prob_match.group(2))]
            continue

        # Match main row
        row_match = row_pattern.search(line)
        if row_match:
            idx = int(row_match.group(1))
            nn_pred = int(row_match.group(2))
            nn_conf = float(row_match.group(3))
            xgb_pred = int(row_match.group(4))
            xgb_conf = float(row_match.group(5))
            true = int(row_match.group(6))
            status = row_match.group(7)

            broken_by_xgb = "BROKEN by XGB" in status

            xgb_prob_true = current_probs[true] if current_probs else np.nan

            confidence_discrepancy = (
                nn_conf - xgb_prob_true if broken_by_xgb else np.nan
            )

            data.append({
                "idx": idx,
                "nn_pred": nn_pred,
                "nn_conf": nn_conf,
                "xgb_pred": xgb_pred,
                "xgb_conf": xgb_conf,
                "true": true,
                "xgb_prob_true": xgb_prob_true,
                "broken_by_xgb": broken_by_xgb,
                "confidence_discrepancy": confidence_discrepancy
            })

    return pd.DataFrame(data)

df = parse_log_file("merged")
df.to_csv("results.csv")
print(df.head())