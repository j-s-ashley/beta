import argparse
import csv
import matplotlib.pyplot as plt

"""
USAGE EXAMPLES
python plot_aucs.py --x n_trees --xlabel "Number of trees in TMVA forest"
python plot_aucs.py --x min_node_size --xlabel "Min node size (%)"
python plot_aucs.py --x max_depth --xlabel "Max depth"
python plot_aucs.py --x min_node_size \
  --filter n_trees=800 --filter max_depth=3 --filter beta=_5 --filter split=50_50 \
  --xlabel "Min node size (%)"
python plot_aucs.py --x n_trees --group beta --xlabel "Number of trees"
"""

MISSING = {"", "None", None}

NUMERIC_FIELDS = {
    "thickness_mu": int,
    "n_trees": int,
    "max_depth": int,
    "min_node_size": float, 
    "train_auc": float,
    "eval_auc": float,
}

# Type handling helper function
def coerce(field, value):
    if value in MISSING:
        return None
    caster = NUMERIC_FIELDS.get(field, None)
    if caster is None:
        return value  # keep as string
    # tolerant parse for int fields that may be written like "5.0"
    if caster is int:
        return int(float(value))
    return caster(value)

def parse_filters(filter_args):
    """
    --filter key=value  (repeatable)
    Values are kept as strings.
    """
    out = {}
    for item in (filter_args or []):
        if "=" not in item:
            raise ValueError(f"Bad --filter '{item}'. Use key=value.")
        k, v = item.split("=", 1)
        out[k] = v
    return out

def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {}
            for key, value in raw_row.items():
                row[key] = coerce(key, value)
            rows.append(row)
    return rows

def apply_filters(rows, filters):
    if not filters:
        return rows
    filtered_rows = []
    for row in rows:
        keep = True
        for key, filter_value in filters.items():
            coerced_filter_value = coerce(key, filter_value)
            if row.get(key) != coerced_filter_value:
                keep = False
                break
        if keep:
            filtered_rows.append(row)
    return filtered_rows

def sort_key(x):
    # numeric sorts naturally; strings sorted lexicographically
    return (0, x) if isinstance(x, (int, float)) else (1, str(x))

def group_rows(rows, group_key):
    grouped = {}
    for row in rows:
        group_value = row.get(group_key)
        if group_value is None:
            continue
        if group_value not in grouped:
            grouped[group_value] = []
        grouped[group_value].append(row)
    return grouped

def plot_series(rows, x_key, y_key, group_key, title, out_png, x_label=None):
    grouped = group_rows(rows, group_key)
    fig, ax = plt.subplots()
    for group_value in sorted(grouped.keys(), key=sort_key):
        points = grouped[group_value]
        xs     = []
        ys     = []

        # collect points
        for p in points:
            x = p.get(x_key)
            y = p.get(y_key)
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)

        # sort points by x
        combined = list(zip(xs, ys))
        combined.sort(key=lambda t: sort_key(t[0]))
        xs_sorted = []
        ys_sorted = []
        for x, y in combined:
            xs_sorted.append(x)
            ys_sorted.append(y)
        ax.plot(xs_sorted, ys_sorted, "-", label=str(group_value))

    ax.set_xlabel(x_label if x_label else x_key)
    ax.set_ylabel("ROC AUC")
    ax.set_title(title)
    ax.legend()

    plt.savefig(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="roc_auc_summary.csv")
    ap.add_argument("--x", default="n_trees", help="column to use for x-axis")
    ap.add_argument("--group", default="thickness_mu", help="column to group curves by")
    ap.add_argument("--filter", action="append", help="repeatable: key=value")
    ap.add_argument("--train", action="store_true", help="make training plot")
    ap.add_argument("--eval", action="store_true", help="make evaluation plot")
    ap.add_argument("--xlabel", default=None, help="override x-axis label")
    args = ap.parse_args()

    rows    = read_rows(args.csv)
    filters = parse_filters(args.filter)
    rows    = apply_filters(rows, filters)

    # default: make both if neither specified
    do_train = args.train or (not args.train and not args.eval)
    do_eval  = args.eval or (not args.train and not args.eval)

    # Nice default group labels for thickness
    group_label = args.group
    if args.group == "thickness_mu":
        # Set legend to "50 microns" etc by overriding label formatting
        pass

    # If grouping by thickness_mu, label it nicely
    if args.group == "thickness_mu":
        for r in rows:
            if r.get("thickness_mu") is not None:
                r["thickness_mu"] = f'{r["thickness_mu"]} microns'

    if do_train:
        plot_series(
            rows,
            args.x,
            "train_auc",
            args.group,
            f"ROC AUCs for Training Data vs {args.x}",
            f"roc-auc-training-vs-{args.x}.png",
            args.xlabel
        )

    if do_eval:
        plot_series(
            rows,
            args.x,
            "eval_auc",
            args.group,
            f"ROC AUCs for Evaluation Data vs {args.x}",
            f"roc-auc-eval-vs-{args.x}.png",
            args.xlabel
        )

if __name__ == "__main__":
    main()
