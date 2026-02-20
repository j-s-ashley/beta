import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cv_aucs.csv")

def plot(x_header, y_header, x_lab, y_lab):
    grid = df.pivot_table(
        index=y_header,
        columns=x_header,
        values="auc",
        aggfunc="mean"
    )
    x_vals = grid.columns.values
    y_vals = grid.index.values
    z      = grid.values

    # Get top AUC values for hyperparameter pair
    flat       = z.ravel()        # flatten for processing
    valid_mask = ~np.isnan(flat)  # mask non-NaN AUCs
    n_valid  = np.count_nonzero(valid_mask)
    if n_valid <5:
        raise ValueError(
            f"{x_lab} vs {y_lab}: fewer than 5 valid AUC entries "
            f"({n_valid} found)"
        )

    valid_indices = np.nonzero(valid_mask)[0]
    valid_vals    = flat[valid_mask]

    top_idx      = np.argpartition(valid_vals, -5)[-5:]           # indices of 5 largest values (unsorted)
    top_idx      = top_idx[np.argsort(valid_vals[top_idx])[::-1]] # sort values
    top_flat_idx = valid_indices[top_idx]                         # map back to original flat indices
    rows, cols = np.unravel_index(top_idx, z.shape)               # unflatten for easy printing

    print(f"\nTop 5 AUCs for {x_lab} vs {y_lab}:")
    for r, c in zip(rows, cols):
        print(
            f"AUC={z[r, c]:.6f}, "
            f"{x_header}={x_vals[c]}, "
            f"{y_header}={y_vals[r]}"
        )


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
