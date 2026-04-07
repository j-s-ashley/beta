import ROOT
import argparse
import numpy as np
from ROOT import TMVA, TFile, TH1F
from sklearn.metrics import roc_curve, auc
from array import array

root_path = "/global/cfs/projectdirs/atlas/jashley/mjolnir/beta"

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

fold = array('i', [0]) # integer spectator (replaces "I" from training)
reader.AddSpectator("fold", fold)

reader.BookMVA("BDT", f"{root_path}/code/datasetcv/weights/cv_job_BDT.weights.xml")

# --- INPUT --- #
sig_file = ROOT.TFile.Open(f"{root_path}/data/MAIA/signal/{sensor_thickness}_sig_eval_ttree.root")
bkg_file = ROOT.TFile.Open(f"{root_path}/data/MAIA/bg/{sensor_thickness}_bkg_eval_ttree.root")
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
fpr, tpr, _ = roc_curve(y_true, y_score)
bkg_rej     = 1 - fpr
sig_eff     = tpr
roc_auc     = auc(sig_eff, bkg_rej)
print(f"Evaluation ROC AUC (Signal efficiency vs Background rejection) = {roc_auc:.3f}")

# --- OUTPUT --- #
kfcv_tag = get_kfcv_tag()
out_file_name = f"{kfcv_tag}_eval_score_data.npz"

eval_sig_scores = np.array(sig_scores)
eval_bkg_scores = np.array(bkg_scores)

np.savez(out_file_name, eval_sig_scores = eval_sig_scores, eval_bkg_scores = eval_bkg_scores)

# Save score histograms
h_sig_eval = ROOT.TH1F("h_sig_eval_score", f"{sensor_thickness} #mum, {n_trees} trees, {min_node_size} minimum node size, {max_depth} maximum depth, {beta} AdaBoost beta;BDT Score;Entries", 100, -1, 1)
for s in eval_sig_scores:
    h_sig_eval.Fill(s)
h_sig_eval_norm = h_sig_eval.Clone("h_sig_eval_norm")
h_sig_eval_norm.Scale(1. / h_sig_eval.Integral())

h_bkg_eval = ROOT.TH1F("h_bkg_eval_score", f"{sensor_thickness} #mum Background BDT Output (Evaluation);BDT Score;Entries", 100, -1, 1)
for b in eval_bkg_scores:
    h_bkg_eval.Fill(b)
h_bkg_eval_norm = h_bkg_eval.Clone("h_bkg_eval_norm")
h_bkg_eval_norm.Scale(1. / h_bkg_eval.Integral())

# Non-normalized histograms
c_hist_eval = ROOT.TCanvas("c_hist_eval", "BDT Output (Non-normalized)", 800, 600)

# Fix y-axis to data max
eval_hists      = [h_sig_eval, h_bkg_eval]
eval_global_max = max(h.GetMaximum() for h in eval_hists)
eval_y_max      = 1.2 * eval_global_max

# Draw pretty plots
h_sig_eval.SetLineColor(ROOT.kRed)
h_bkg_eval.SetLineColor(ROOT.kBlue)
h_sig_eval.SetLineWidth(2)
h_bkg_eval.SetLineWidth(2)
h_sig_eval.SetFillColorAlpha(ROOT.kRed, 0.35)
h_bkg_eval.SetFillColorAlpha(ROOT.kBlue, 0.35)
h_sig_eval.SetMaximum(eval_y_max)
h_sig_eval.Draw("HIST")
h_bkg_eval.Draw("HIST SAME")
h_sig_eval.SetStats(0)

# Add a legend
eval_hist_legend = ROOT.TLegend(0.7, 0.75, 0.85, 0.85)
eval_hist_legend.AddEntry(h_sig_eval, "Signal clusters", "l")
eval_hist_legend.AddEntry(h_bkg_eval, "Background clusters", "l")
eval_hist_legend.SetBorderSize(0)
eval_hist_legend.Draw()

c_hist_eval.SaveAs(f"{kfcv_tag}_BDT_evaluation_of_eval_wpixels.png")
c_hist_eval.SaveAs(f"{kfcv_tag}_BDT_evaluation_of_eval_wpixels.pdf")
c_hist_eval.Write()

# Normalized histograms
c_hist_eval_norm = ROOT.TCanvas("c_hist_eval_norm", f"BDT Output", 800, 600)

# Fix y-axis to data max
eval_norm_hists      = [h_sig_eval_norm, h_bkg_eval_norm]
eval_norm_global_max = max(h.GetMaximum() for h in eval_norm_hists)
eval_norm_y_max      = 1.2 * eval_norm_global_max

# Draw pretty plots
h_sig_eval_norm.SetLineColor(ROOT.kRed)
h_bkg_eval_norm.SetLineColor(ROOT.kBlue)
h_sig_eval_norm.SetLineWidth(2)
h_bkg_eval_norm.SetLineWidth(2)
h_sig_eval.SetFillColorAlpha(ROOT.kRed, 0.35)
h_bkg_eval.SetFillColorAlpha(ROOT.kBlue, 0.35)
h_sig_eval_norm.SetMaximum(eval_norm_y_max)
h_sig_eval_norm.Draw("HIST")
h_bkg_eval_norm.Draw("HIST SAME")
h_sig_eval_norm.SetStats(0)

# Add a legend
eval_hist_legend = ROOT.TLegend(0.7, 0.75, 0.85, 0.85)
eval_hist_legend.AddEntry(h_sig_eval_norm, "Signal clusters", "l")
eval_hist_legend.AddEntry(h_bkg_eval_norm, "Background clusters", "l")
eval_hist_legend.SetBorderSize(0)
eval_hist_legend.Draw()

c_hist_eval_norm.SaveAs(f"{kfcv_tag}_BDT_evaluation_of_eval_wpixels.png")
c_hist_eval_norm.SaveAs(f"{kfcv_tag}_BDT_evaluation_of_eval_wpixels.pdf")
c_hist_eval_norm.Write()

# ROC curve
c_roc_eval = ROOT.TCanvas("c_roc_eval", "{sensor_thickness} #mum, {n_trees} trees, {min_node_size} minimum node size, {max_depth} maximum depth, {beta} AdaBoost beta", 600, 600)
g_eval = ROOT.TGraph(len(sig_eff), array('f', bkg_rej), array('f', sig_eff))

eval_roc_legend = ROOT.TLegend(0.15, 0.20, 0.35, 0.35)
g_eval.SetTitle(f"{sensor_thickness} #mum, {n_trees} trees, {min_node_size} minimum node size, {max_depth} maximum depth, {beta} AdaBoost beta;Background Rejection;Signal Efficiency")
g_eval.SetLineColor(ROOT.kBlue)
g_eval.SetLineWidth(2)
g_eval.Draw("AL")
g_eval.GetXaxis().SetLimits(0, 1.1)
g_eval.GetYaxis().SetRangeUser(0, 1.1)

c_roc_eval.SaveAs(f"{kfcv_tag}_BDT_ROC_of_eval_SigEff_vs_BkgRej_wpixels.png")
c_roc_eval.SaveAs(f"{kfcv_tag}_BDT_ROC_of_eval_SigEff_vs_BkgRej_wpixels.pdf")
g_eval.Write("eval_ROC_curve")
