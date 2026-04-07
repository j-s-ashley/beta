import csv
import math
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

"""
Read cv_aucs.csv and make a box plot comparing the distribution of AUC values
(across folds) for selected hyperparameter combinations:
(thickness, k, n_trees, min_node_size, max_depth, beta).

Expected columns:
thickness,k,n_trees,min_node_size,max_depth,beta,fold,auc,auc_avg,auc_std
"""

# --- HYPERPARAMETER COMBINATIONS --- #
# Each combo is a dict specifying the first 6 columns
# IMPORTANT: match types to CSV (ints/floats/strings)
COMBOS = [
    {"thickness": 50, "k": 5, "n_trees": 400, "min_node_size": "1.5%", "max_depth": 14,  "beta": 0.1},
    {"thickness": 50, "k": 5, "n_trees": 400, "min_node_size": "1.0%", "max_depth": 14,  "beta": 0.1},
    {"thickness": 50, "k": 5, "n_trees": 600, "min_node_size": "1.5%", "max_depth": 14,  "beta": 0.1},
    {"thickness": 50, "k": 5, "n_trees": 400, "min_node_size": "0.5%", "max_depth": 14,  "beta": 0.1},
    {"thickness": 50, "k": 5, "n_trees": 400, "min_node_size": "1.0%", "max_depth": 13,  "beta": 0.1},
]

CSV_PATH = "cv_aucs.csv"
OUT_PNG  = "auc_boxplot.png"

TITLE    = "AUC distribution across CV folds by hyperparameter combination"
Y_LABEL  = "AUC"

# Helpers --- #
FIRST6 = ["thickness", "k", "n_trees", "min_node_size", "max_depth", "beta"]

def _to_number_if_possible(s):
    """Convert string to int/float when safe; else return original string."""
    s2 = str(s).strip()
    if s2 == "":
        return s2
    try:
        if any(c in s2 for c in (".", "e", "E")):
            return float(s2)
        return int(s2)
    except ValueError:
        try:
            return float(s2)
        except ValueError:
            return s2

def _values_match(row_val, combo_val, float_tol=1e-12):
    """Equality with float tolerance."""
    # Normalize both sides to numbers when possible
    rv = _to_number_if_possible(row_val)
    cv = combo_val
    # If either is float, compare with tolerance
    if isinstance(rv, float) or isinstance(cv, float):
        try:
            return math.isclose(float(rv), float(cv), rel_tol=0.0, abs_tol=float_tol)
        except Exception:
            return False
    return rv == cv

def combo_key(combo):
    """Canonical immutable key for dicts."""
    return tuple((k, combo[k]) for k in FIRST6)

def combo_label(combo):
    """Pretty label shown on x-axis."""
    return f"NTrees={combo['n_trees']}, MinNodeSize={combo['min_node_size']}, MaxDepth={combo['max_depth']}, beta={combo['beta']}"

def read_auc_by_combo(csv_path, combos):
    """
    Returns:
      labels: list[str]
      auc_lists: list[list[float]] in same order as combos
    """
    # Prepare lookup
    combos_by_key = OrderedDict()
    for c in combos:
        combos_by_key[combo_key(c)] = c

    aucs = {k: [] for k in combos_by_key.keys()}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # sanity check
        for req in (FIRST6 + ["auc"]):
            if req not in reader.fieldnames:
                raise ValueError(f"Missing required column '{req}' in {csv_path}. Found: {reader.fieldnames}")

        for row in reader:
            for ck, combo in combos_by_key.items():
                ok = True
                for col in FIRST6:
                    if not _values_match(row[col], combo[col]):
                        ok = False
                        break
                if ok:
                    aucs[ck].append(float(row["auc"]))

    labels = [combo_label(combos_by_key[k]) for k in combos_by_key.keys()]
    auc_lists = [aucs[k] for k in combos_by_key.keys()]
    return labels, auc_lists

# --- Plot stuff --- #
def plot_stuff(labels, auc_lists, out_png):
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(auc_lists, labels=labels, showmeans=False, patch_artist=True)

    for box in bp["boxes"]:
        box.set(facecolor="#6ea8fe", alpha=0.35, edgecolor="black", linewidth=1.0)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2.0)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.0)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.0)

    ax.set_title(TITLE)
    ax.set_ylabel(Y_LABEL)

    flat = [x for vals in auc_lists for x in vals]
    if flat:
        ax.set_ylim(min(flat) - 0.001, max(flat) + 0.001)
    else:
        ax.set_ylim(0.0, 1.0)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Saved plot to {out_png}")

# --- Main --- #
def main():
    labels, auc_lists = read_auc_by_combo(CSV_PATH, COMBOS)

    # Basic reporting
    for lab, vals in zip(labels, auc_lists):
        print(f"{lab}: n={len(vals)}")
        print(vals)

    plot_stuff(labels, auc_lists, OUT_PNG)

if __name__ == "__main__":
    main()
