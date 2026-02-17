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
