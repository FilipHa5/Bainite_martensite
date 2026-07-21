import glob
import re
import ast
from collections import Counter, defaultdict
import statistics

DIR = "ttest"

nn_pattern = re.compile(r"Best inner hyperparams NN:\s*(\{.*?\})")
dt_pattern = re.compile(r"Best inner hyperparams DT:\s*(\{.*?\})")
acc_pattern = re.compile(r"accuracy score:\s*([0-9.]+)")

records = []


path = f"{DIR}/*.out"

for fname in glob.glob(path):
    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    nns = nn_pattern.findall(text)
    dts = dt_pattern.findall(text)
    accs = acc_pattern.findall(text)

    n = min(len(nns), len(dts), len(accs))

    for nn, dt, acc in zip(nns[:n], dts[:n], accs[:n]):
        nn_dict = ast.literal_eval(nn)
        dt_dict = ast.literal_eval(dt)
        acc = float(acc)

        records.append({
            "file": fname,
            "nn": nn_dict,
            "dt": dt_dict,
            "acc": acc
        })

print(f"Found {len(records)} nested-CV results.\n")

# -------------------------
# Frequency tables
# -------------------------
nn_counter = Counter()
dt_counter = Counter()

combo_acc = defaultdict(list)

for r in records:
    nn_key = tuple(sorted(r["nn"].items()))
    dt_key = tuple(sorted(r["dt"].items()))

    nn_counter[nn_key] += 1
    dt_counter[dt_key] += 1
    combo_acc[(nn_key, dt_key)].append(r["acc"])

print("NN hyperparameter frequencies")
print("-" * 60)
for hp, cnt in nn_counter.most_common():
    print(dict(hp), f"selected {cnt} times")

print("\nDT hyperparameter frequencies")
print("-" * 60)
for hp, cnt in dt_counter.most_common():
    print(dict(hp), f"selected {cnt} times")

print("\nCombined configuration performance")
print("-" * 60)

summary = []

for (nn, dt), accs in combo_acc.items():
    summary.append({
        "nn": dict(nn),
        "dt": dict(dt),
        "count": len(accs),
        "mean": statistics.mean(accs),
        "std": statistics.stdev(accs) if len(accs) > 1 else 0,
        "max": max(accs)
    })

summary.sort(key=lambda x: (-x["mean"], -x["count"]))

for s in summary:
    print(f"Count={s['count']:2d} "
          f"Mean={s['mean']:.4f} "
          f"Std={s['std']:.4f} "
          f"Max={s['max']:.4f}")
    print("  NN:", s["nn"])
    print("  DT:", s["dt"])
    print()

overall = max(records, key=lambda r: r["acc"])

print("=" * 60)
print("OVERALL BEST SINGLE RESULT")
print("=" * 60)
print("Accuracy:", overall["acc"])
print("File:", overall["file"])
print("NN:", overall["nn"])
print("DT:", overall["dt"])

print("\nOVERALL ACCURACY")
print(f"Mean : {statistics.mean(r['acc'] for r in records):.4f}")
print(f"Std  : {statistics.stdev(r['acc'] for r in records):.4f}")
print(f"Max  : {max(r['acc'] for r in records):.4f}")
print(f"Min  : {min(r['acc'] for r in records):.4f}")

print("\nRECOMMENDED CONFIGURATION")
print(summary[0]["nn"])
print(summary[0]["dt"])
print(f"Average accuracy = {summary[0]['mean']:.4f}")
print(f"Selected {summary[0]['count']} times")