import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cv_aucs.csv")

def plot(x_header, y_header, x_lab, y_lab):
    grid = df.pivot_table(
        index=y_header,
        columns=x_header,
        values="auc_avg",
        aggfunc="max"
    )
    x_vals = grid.columns.values
    y_vals = grid.index.values
    z      = grid.values

    # Get top AUC values for hyperparameter pair
    max_per_group = ( 
        df.groupby([x_header, y_header],    # group by avg AUC val
            as_index=False)["auc_avg"]      # remove duplicate avgs per x,y pair
        .max()                              # keep only max per x,y pair
    )

    max_per_group = max_per_group.dropna(subset=["auc_avg"]) # no blanks
    if len(max_per_group) < 5:
        raise ValueError("Fewer than 5 valid scores. Check input.")

    top_scores = max_per_group.sort_values(                  # sort and get top 5
        "auc_avg", ascending=False).head(5) 
    top_scores = top_scores.reset_index(drop=True)           # overwrite original indices

    print(f"\nTop 5 AUCs for {x_lab} vs {y_lab}:")
    print(top_scores)

    plt.figure()

    mesh = plt.pcolormesh(x_vals, y_vals, z, shading="auto")
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.colorbar(mesh, label="ROC AUC")

    plt.savefig(f"{x_lab}-vs-{y_lab}-aucs.pdf")

plot("min_node_size", "max_depth", "MinNodeSize", "MaxDepth")
plot("min_node_size", "beta", "MinNodeSize", "Beta")
plot("min_node_size", "n_trees", "MinNodeSize", "NTrees")
plot("max_depth", "beta", "MaxDepth", "Beta")
plot("max_depth", "n_trees", "MaxDepth", "NTrees")
plot("n_trees", "beta", "NTrees", "Beta")
