import ROOT
import argparse
import numpy as np
from ROOT import TMVA, TFile, TH1F
from sklearn.metrics import roc_curve, auc
from array import array

def options():
    parser = argparse.ArgumentParser(
        description="Train BDT on data from input TTree files."
    )   
    parser.add_argument(
        "-k", required=True, type=int, help="Number of folds"
    )   
    parser.add_argument(
        "-t", required=True, type=int, help="VXB sensor thickness"
    )   
    parser.add_argument(
        "-s", required=True, type=str, help="MinNodeSize"
    )   
    parser.add_argument(
        "-d", required=True, type=int, help="MaxDepth"
    )   
    parser.add_argument(
        "-n", required=True, type=int, help="NTrees"
    )   
    parser.add_argument(
        "-b", required=True, type=str, help="AdaBoost beta"
    )   
    return parser.parse_args()

k_folds          = options().k
sensor_thickness = options().t
min_node_size    = options().s
max_depth        = options().d
n_trees          = options().n
beta             = options().b

def get_kfcv_tag():
    tag = (
        str(sensor_thickness) + "_" +
        str(min_node_size[0]) + "-" +
        str(min_node_size[2]) + "_" +
        str(max_depth) + "_" +
        str(n_trees) + "_" +
        str(beta[0]) + "-" +
        str(beta[2])
    )
    return tag

ROOT.TMVA.Tools.Instance()
reader    = ROOT.TMVA.Reader("!Color:Silent")
variables = [
    "Cluster_ArrivalTime",
    "Cluster_EnergyDeposited",
    "Incident_Angle",
    "Cluster_Size_x",
    "Cluster_Size_y",
    "Cluster_Size_tot",
    "Cluster_x",
    "Cluster_y",
    "Cluster_z",
    "Cluster_RMS_x",
    "Cluster_RMS_y",
    "Cluster_Skew_x",
    "Cluster_Skew_y",
    "Cluster_AspectRatio",
    ]

# NOTE: The following two for loops must be kept separate to preserve variable order
for i in range(9):
    variables.append(f"PixelHits_EnergyDeposited_{i}")

for i in range(9):
    variables.append(f"PixelHits_ArrivalTime_{i}")

buffers = {v: array('f', [0.]) for v in variables}
for v in variables:
    reader.AddVariable(v, buffers[v])

reader.BookMVA("BDT", "datasetcv/weights/TMVAClassification_BDT.weights.xml")

# --- INPUT --- #
sig_file = ROOT.TFile.Open(f"../data/MAIA/signal/{sensor_thickness}_sig_ttree.root")
bkg_file = ROOT.TFile.Open(f"../data/MAIA/bg/{sensor_thickness}_bkg_ttree.root")
sig_tree = sig_file.Get("HitTree")
bkg_tree = bkg_file.Get("HitTree")

def evaluate_flat_tree(flat_tree, scores_list):
    for evt in flat_tree:
        for v in variables:
            buffers[v][0] = getattr(evt, v)
        score = reader.EvaluateMVA("BDT")
        scores_list.append(score)

# --- EVALUATE SIGNAL AND BACKGROUND --- #
sig_scores = []
bkg_scores = []
evaluate_flat_tree(sig_tree, sig_scores)
evaluate_flat_tree(bkg_tree, bkg_scores)

y_true      = np.array([1]*len(sig_scores) + [0]*len(bkg_scores))
y_score     = np.array(sig_scores + bkg_scores)
fpr, tpr,  = roc_curve(y_true, y_score)
bkg_rej     = 1 - fpr
sig_eff     = tpr
roc_auc     = auc(sig_eff, bkg_rej)
print(f"Evaluation ROC AUC (Signal efficiency vs Background rejection) = {roc_auc:.3f}")

# --- OUTPUT --- #
out_file_name = f"{get_kfcv_tag()}_eval_score_data.npz"

eval_sig_scores = np.array(sig_scores)
eval_bkg_scores = np.array(bkg_scores)

np.savez(out_file_name, eval_sig_scores = eval_sig_scores, eval_bkg_scores = eval_bkg_scores)
