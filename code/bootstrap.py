import csv
import ROOT
import argparse
import numpy as np
from ROOT import TMVA, TFile, TH1F
from sklearn.metrics import roc_curve, auc
from array import array

def options():
    parser = argparse.ArgumentParser(description="Train BDT on data from input TTree files.")
    parser.add_argument("-t", required=True, type=int, help="VXB sensor thickness")
    parser.add_argument("-n", required=True, type=int, help="Number of resamples")
    parser.add_argument("-s", required=False, type=int, help="Random generator seed")
    return parser.parse_args()

sensor_thickness = options().t
num_resample     = options().n
generator_seed   = options().s

def evaluate_flat_tree(flat_tree, scores_list):
    for evt in flat_tree:
        for v in variables:
            buffers[v][0] = getattr(evt, v)
        score = reader.EvaluateMVA("BDT")
        scores_list.append(score)

def bootstrap_auc(sig_scores, bkg_scores, set_name):
    rng   = np.random.default_rng(generator_seed)
    n_sig = len(sig_scores)
    n_bkg = len(bkg_scores)
    aucs  = np.empty(num_resample)

    for i in range(num_resample):
        sig_idx = rng.integers(0, n_sig, n_sig)
        bkg_idx = rng.integers(0, n_bkg, n_bkg)
        scores = np.concatenate([sig_scores[sig_idx], bkg_scores[bkg_idx]])

    y_true = np.array([1]*len(sig_scores) + [0]*len(bkg_scores))
    y_score     = np.array(sig_scores + bkg_scores)
    fpr, tpr, set_  = roc_curve(y_true, y_score)
    bkg_rej     = 1 - fpr
    sig_eff     = tpr
    roc_auc     = auc(sig_eff, bkg_rej)

    print(f"{set_name} signal scores:")
    print(sig_scores)
    print(f"{set_name} background scores:")
    print(bkg_scores)

    return roc_auc

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

reader.BookMVA("BDT", "dataset/weights/TMVAClassification_BDT.weights.xml")

# --- INPUT --- #
# Training files
sig_training_file = ROOT.TFile.Open(f"../data/MAIA/signal/{sensor_thickness}_sig_trng_ttree.root")
bkg_training_file = ROOT.TFile.Open(f"../data/MAIA/bg/{sensor_thickness}_bkg_trng_ttree.root")
sig_training_tree = sig_training_file.Get("HitTree")
bkg_training_tree = bkg_training_file.Get("HitTree")
# Evaluation files
sig_eval_file = ROOT.TFile.Open(f"../data/MAIA/signal/{sensor_thickness}_sig_eval_ttree.root")
bkg_eval_file = ROOT.TFile.Open(f"../data/MAIA/bg/{sensor_thickness}_bkg_eval_ttree.root")
sig_eval_tree = sig_eval_file.Get("HitTree")
bkg_eval_tree = bkg_eval_file.Get("HitTree")

# --- EVALUATE SIGNAL AND BACKGROUND --- #
# --- Training evaluation --- #
sig_training_scores = []
bkg_training_scores = []
evaluate_flat_tree(sig_training_tree, sig_training_scores)
evaluate_flat_tree(bkg_training_tree, bkg_training_scores)
training_auc = bootstrap_auc(sig_training_scores, bkg_training_scores, "Training")
print(f"Training AUC = {training_auc:.3f}")

# --- Evaluation evaluation --- #
sig_eval_scores = []
bkg_eval_scores = []
evaluate_flat_tree(sig_eval_tree, sig_eval_scores)
evaluate_flat_tree(bkg_eval_tree, bkg_eval_scores)
evaluation_auc = bootstrap_auc(sig_eval_scores, bkg_eval_scores, "Evaluation")
print(f"Evaluation AUC = {evaluation_auc:.3f}")
