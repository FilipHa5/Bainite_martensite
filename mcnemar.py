from pathlib import Path
import re
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

LOG_DIR = Path("tuned_thr")

# Match each sample block
sample_pattern = re.compile(
    r"\[(\d+)\](.*?)(?=\n\[\d+\]|\Z)",
    re.DOTALL
)

rows = []

for f in sorted(LOG_DIR.glob("*.out")):

    text = f.read_text(errors="ignore")

    fixes = 0
    breaks = 0
    used = 0

    for _, block in sample_pattern.findall(text):

        if "Decision=Override" not in block:
            continue

        used += 1

        if "Status=✅ FIX" in block:
            fixes += 1

        elif "Status=❌ BREAK" in block:
            breaks += 1

    rows.append({
        "File": f.name,
        "Fixes": fixes,
        "Breaks": breaks,
        "XGB_used": used,
        "Net_gain": fixes - breaks,
        "Fix_rate": fixes / used if used else 0,
        "Break_rate": breaks / used if used else 0,
    })

df = pd.DataFrame(rows)

print("\n========== PER FILE ==========")
print(df)

print("\n========== SUMMARY ==========")

summary = {
    "Files": len(df),
    "Total fixes": df.Fixes.sum(),
    "Total breaks": df.Breaks.sum(),
    "Total overrides": df.XGB_used.sum(),
    "Net gain": df.Net_gain.sum(),
    "Fix rate": df.Fixes.sum() / df.XGB_used.sum(),
    "Break rate": df.Breaks.sum() / df.XGB_used.sum(),
}

for k, v in summary.items():
    print(f"{k:20s}: {v}")

# McNemar
table = [
    [0, df.Breaks.sum()],
    [df.Fixes.sum(), 0]
]

result = mcnemar(table, exact=True)

print("\n========== MCNEMAR ==========")
print(f"Statistic : {result.statistic}")
print(f"p-value   : {result.pvalue:.6g}")

if result.pvalue < 0.05:
    print("Difference is statistically significant.")
else:
    print("No statistically significant difference.")

df.to_csv(LOG_DIR / "hybrid_error_analysis.csv", index=False)
