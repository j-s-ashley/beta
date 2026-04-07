import csv
import math
import os
import sys
import ROOT
from collections import defaultdict, OrderedDict

"""
Read cv_aucs.csv and make a box plot comparing the distribution of AUC values
(across folds) for selected hyperparameter combinations:
(thickness, k, n_trees, min_node_size, max_depth, beta).

Expected columns:
thickness,k,n_trees,min_node_size,max_depth,beta,fold,auc,auc_avg,auc_std
"""

# -----------------------------
# HYPERPARAMETER COMBINATIONS
# -----------------------------
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
OUT_PDF  = "auc_boxplot.pdf"

TITLE    = "AUC distribution across CV folds by hyperparameter combination"
Y_LABEL  = "AUC"

# -----------------------------
# Helpers
# -----------------------------
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
def plot_stuff(labels, auc_lists, out_pdf):
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    n = len(labels)
    if n == 0:
        raise ValueError("No combinations provided.")
    if any(len(v) == 0 for v in auc_lists):
        print("WARNING: Some combinations have 0 matching rows; they will appear empty.", file=sys.stderr)

    c = ROOT.TCanvas("c", "c", 1200, 600)
    c.SetLeftMargin(0.07)
    c.SetRightMargin(0.02)
    c.SetBottomMargin(0.28)
    c.SetTopMargin(0.10)

    # Compute quantiles per combo, then draw TBox + TLine.
    frame = ROOT.TH2F("frame", TITLE, n, 0.5, n + 0.5, 10, 0.97, 1.01)
    frame.GetYaxis().SetTitle(Y_LABEL)
    frame.GetXaxis().SetLabelSize(0.03)
    frame.GetXaxis().SetTitleSize(0.04)
    frame.GetYaxis().SetTitleSize(0.04)
    frame.GetXaxis().SetNdivisions(n, False)

    for i, lab in enumerate(labels, start=1):
        frame.GetXaxis().SetBinLabel(i, lab)

    frame.Draw("AXIS")

    def quantiles(vals):
        v = sorted(vals)
        if len(v) == 0:
            return None
        def q(p):
            # linear interpolation
            pos = p * (len(v) - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return v[lo]
            return v[lo] * (hi - pos) + v[hi] * (pos - lo)
        return {
            "min": v[0],
            "q1": q(0.25),
            "med": q(0.50),
            "q3": q(0.75),
            "max": v[-1],
        }

    box_half_width = 0.25
    debug_offset   = 0.01 # debug
    for i, vals in enumerate(auc_lists, start=1):
        qs = quantiles(vals)
        if qs is None:
            print(f"\nQuantile calc failed for data index {i}") # debug
            continue

        print(f"\nQuantiles for data index {i}")
        print(qs)

        x1, x2 = i - box_half_width, i + box_half_width
        # Box (Q1-Q3)
        box = ROOT.TBox(x1, qs["q1"]+debug_offset, x2, qs["q3"]+debug_offset) # debug
        debug_offset += 0.005
        #box = ROOT.TBox(x1, qs["q1"], x2, qs["q3"])
        box.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
        box.SetLineColor(ROOT.kBlack)
        box.Draw("l f SAME")

        # Median
        med = ROOT.TLine(x1, qs["med"], x2, qs["med"])
        med.SetLineWidth(2)
        med.Draw("SAME")

        # Whiskers
        w1 = ROOT.TLine(i, qs["min"], i, qs["q1"])
        w2 = ROOT.TLine(i, qs["q3"], i, qs["max"])
        w1.Draw("SAME")
        w2.Draw("SAME")

        # Caps
        cap1 = ROOT.TLine(i - 0.15, qs["min"], i + 0.15, qs["min"])
        cap2 = ROOT.TLine(i - 0.15, qs["max"], i + 0.15, qs["max"])
        cap1.Draw("SAME")
        cap2.Draw("SAME")

        c.Modified()
        c.Update()

    c.SaveAs(out_pdf)
    print(f"Saved PyROOT plot to {out_pdf}")

# -----------------------------
# Main
# -----------------------------
def main():
    labels, auc_lists = read_auc_by_combo(CSV_PATH, COMBOS)

    # Basic reporting
    for lab, vals in zip(labels, auc_lists):
        print(f"{lab}: n={len(vals)}")
        print(vals)

    plot_stuff(labels, auc_lists, OUT_PDF)

if __name__ == "__main__":
    main()
