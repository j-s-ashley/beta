import re
import csv
from pathlib import Path

SEARCH_ROOT = Path("results")
OUT_CSV = Path("roc_auc_summary.csv")

RE_TRAIN = re.compile(r"Training\s+ROC AUC.*=\s*([0-9]*\.?[0-9]+)")
RE_EVAL  = re.compile(r"Evaluation\s+ROC AUC.*=\s*([0-9]*\.?[0-9]+)")

# filenames: "50_evaluation.log", "75_evaluation.log", ...
RE_THICK_FILE = re.compile(r"^([0-9]{2,4})_evaluation\.log$", re.IGNORECASE)

def parse_aucs(text: str):
    mt = RE_TRAIN.search(text)
    me = RE_EVAL.search(text)
    train = float(mt.group(1)) if mt else None
    eval_ = float(me.group(1)) if me else None
    return train, eval_

def parse_percent(token):
    if token is None:
        return None
    try:
        return float(token.replace("_", "."))
    except ValueError:
        return None

def parse_subdir(subdir_name: str):
    """
    subdir_name example: "200-5-50_50-_5-3"
    convention: (nTrees)-(min node size)-(split)-(beta)-(max depth)
    """
    parts = subdir_name.split("-")
    if not parts:
        return None, None, None, None, None

    def to_int(x):
        try: return int(x)
        except Exception: return None

    n_trees = to_int(parts[0])
    min_node_size = parse_percent(parts[1]) if len(parts) > 1 else None
    split = parts[2] if len(parts) > 2 else None
    beta = parts[3] if len(parts) > 3 else None
    max_depth = to_int(parts[4]) if len(parts) > 4 else None

    return n_trees, min_node_size, split, beta, max_depth

def main():
    rows = []

    # Each immediate child of results/ is one run configuration dir
    for run_dir in sorted([p for p in SEARCH_ROOT.iterdir() if p.is_dir()]):
        n_trees, min_node_size, split, beta, max_depth = parse_subdir(run_dir.name)

        for log_path in sorted(run_dir.glob("*_evaluation.log")):
            m = RE_THICK_FILE.match(log_path.name)
            if not m:
                continue
            thickness_mu = int(m.group(1))

            text = log_path.read_text(errors="replace")
            train_auc, eval_auc = parse_aucs(text)
            if train_auc is None and eval_auc is None:
                continue

            rows.append({
                "thickness_mu": thickness_mu,
                "n_trees": n_trees,
                "min_node_size": min_node_size,
                "split": split,
                "beta": beta,
                "max_depth": max_depth,
                "train_auc": train_auc,
                "eval_auc": eval_auc,
                "log_path": str(log_path),
                "run_dir": str(run_dir),
            })

    fieldnames = [
        "thickness_mu","n_trees","min_node_size","split","beta","max_depth",
        "train_auc","eval_auc","run_dir","log_path"
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")

    # basic QA
    bad = [r for r in rows if r["n_trees"] is None]
    if bad:
        print(f"WARNING: {len(bad)} rows had n_trees=None (unexpected subdir names).")
        print("Example:", bad[0]["run_dir"])

if __name__ == "__main__":
    main()
