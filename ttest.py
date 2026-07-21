import re
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel

LOG_DIR = Path("ttest2")

patterns = {
    "NN": re.compile(r"Outer test score NN:\s*([0-9.]+)"),
    "DT": re.compile(r"Outer test score secondary:\s*([0-9.]+)"),
    "Hybrid": re.compile(r"Hybrid classification accuracy:\s*([0-9.]+)")
}

rows = []

files = sorted(list(LOG_DIR.glob("*.txt")) + list(LOG_DIR.glob("*.out")))

if not files:
    raise RuntimeError("No log files found.")

for logfile in files:
    text = logfile.read_text(errors="ignore")

    nn = patterns["NN"].findall(text)
    dt = patterns["DT"].findall(text)
    hy = patterns["Hybrid"].findall(text)

    if not (len(nn) == len(dt) == len(hy)):
        raise RuntimeError(
            f"{logfile.name}: NN={len(nn)}, DT={len(dt)}, Hybrid={len(hy)}"
        )

    for seed, (n, d, h) in enumerate(zip(nn, dt, hy), start=1):
        rows.append({
            "File": logfile.name,
            "Seed": seed,
            "NN": float(n),
            "DT": float(d),
            "Hybrid": float(h),
        })

df = pd.DataFrame(rows)

# Save all extracted results
df.to_csv(LOG_DIR / "outer_summary.csv", index=False)

# Average within each outer fold (file)
fold_df = (
    df.groupby("File")[["NN", "DT", "Hybrid"]]
      .mean()
      .reset_index()
)

fold_df.to_csv(LOG_DIR / "outer_fold_means.csv", index=False)

pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print("\n========== PER-SEED RESULTS ==========")
print(df)

print("\n========== PER-FOLD MEANS ==========")
print(fold_df)

print("\n========== SUMMARY (Fold Means) ==========")
summary = pd.DataFrame({
    "Mean": fold_df[["NN", "DT", "Hybrid"]].mean(),
    "Std": fold_df[["NN", "DT", "Hybrid"]].std(),
    "N": fold_df[["NN", "DT", "Hybrid"]].count()
})
print(summary)

print("\n========== PAIRED T-TESTS (Fold Means) ==========")

for a, b in [("NN", "DT"), ("NN", "Hybrid"), ("DT", "Hybrid")]:
    t, p = ttest_rel(fold_df[a], fold_df[b])

    print(f"\n{a} vs {b}")
    print(f"Mean difference : {(fold_df[a]-fold_df[b]).mean():.6f}")
    print(f"t statistic     : {t:.6f}")
    print(f"p value         : {p:.6g}")

print("\nSaved:")
print(LOG_DIR / "outer_summary.csv")
print(LOG_DIR / "outer_fold_means.csv")

# import re
# from pathlib import Path

# import pandas as pd
# from scipy.stats import ttest_rel

# # Directory containing log files
# LOG_DIR = Path("ttest")

# # Regex patterns
# patterns = {
#     "NN": re.compile(r"Outer test score NN:\s*([0-9.]+)"),
#     "DT": re.compile(r"Outer test score secondary:\s*([0-9.]+)"),
#     "Hybrid": re.compile(r"Hybrid classification accuracy:\s*([0-9.]+)")
# }

# rows = []

# # Read every .txt and .out file
# log_files = sorted(list(LOG_DIR.glob("*.txt")) + list(LOG_DIR.glob("*.out")))

# if not log_files:
#     raise RuntimeError(f"No .txt or .out files found in {LOG_DIR}")

# for logfile in log_files:
#     text = logfile.read_text(errors="ignore")

#     nn_scores = patterns["NN"].findall(text)
#     dt_scores = patterns["DT"].findall(text)
#     hybrid_scores = patterns["Hybrid"].findall(text)

#     if not (len(nn_scores) == len(dt_scores) == len(hybrid_scores)):
#         raise RuntimeError(
#             f"{logfile.name}: found "
#             f"{len(nn_scores)} NN, "
#             f"{len(dt_scores)} DT, "
#             f"{len(hybrid_scores)} Hybrid results."
#         )

#     for idx, (nn, dt, hy) in enumerate(
#         zip(nn_scores, dt_scores, hybrid_scores), start=1
#     ):
#         rows.append(
#             {
#                 "File": logfile.name,
#                 "Result": idx,
#                 "NN": float(nn),
#                 "DT": float(dt),
#                 "Hybrid": float(hy),
#             }
#         )

# df = pd.DataFrame(rows)

# if df.empty:
#     raise RuntimeError("No results were extracted from the log files.")

# # Save CSV
# out_csv = LOG_DIR / "outer_summary.csv"
# df.to_csv(out_csv, index=False)

# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 150)
# pd.set_option("display.float_format", lambda x: f"{x:.6f}")

# print("\n========== EXTRACTED RESULTS ==========")
# print(df)

# print("\n========== SUMMARY ==========")
# summary = pd.DataFrame({
#     "Mean": df[["NN", "DT", "Hybrid"]].mean(),
#     "Std": df[["NN", "DT", "Hybrid"]].std(),
#     "Min": df[["NN", "DT", "Hybrid"]].min(),
#     "Max": df[["NN", "DT", "Hybrid"]].max(),
#     "N": df[["NN", "DT", "Hybrid"]].count(),
# })
# print(summary)

# print("\n========== PAIRED T-TESTS ==========")

# comparisons = [
#     ("NN", "DT"),
#     ("NN", "Hybrid"),
#     ("DT", "Hybrid"),
# ]

# for a, b in comparisons:
#     t_stat, p_val = ttest_rel(df[a], df[b])

#     print(f"\n{a} vs {b}")
#     print(f"Mean {a}:           {df[a].mean():.6f}")
#     print(f"Mean {b}:           {df[b].mean():.6f}")
#     print(f"Mean difference:    {(df[a] - df[b]).mean():.6f}")
#     print(f"t statistic:        {t_stat:.6f}")
#     print(f"p value:            {p_val:.6g}")

# print(f"\nProcessed {len(log_files)} files.")
# print(f"Extracted {len(df)} result sets.")
# print(f"CSV written to: {out_csv}")