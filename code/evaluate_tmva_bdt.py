import ROOT
import argparse
import numpy as np
from ROOT import TMVA, TFile, TH1F
from sklearn.metrics import roc_curve, auc
from array import array

def options():
    parser = argparse.ArgumentParser(description="Train BDT on data from input TTree files.")
    parser.add_argument("-t", required=True, type=int, help="VXB sensor thickness")
    return parser.parse_args()

sensor_thickness = options().t

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
sig_training_file = ROOT.TFile.Open(f"../../data/beta/MAIA/signal/{sensor_thickness}_sig_trng_ttree.root")
bkg_training_file = ROOT.TFile.Open(f"../../data/beta/MAIA/bg/{sensor_thickness}_bkg_trng_ttree.root")
sig_training_tree = sig_training_file.Get("HitTree")
bkg_training_tree = bkg_training_file.Get("HitTree")
# Evaluation files
sig_eval_file = ROOT.TFile.Open(f"../../data/beta/MAIA/signal/{sensor_thickness}_sig_eval_ttree.root")
bkg_eval_file = ROOT.TFile.Open(f"../../data/beta/MAIA/bg/{sensor_thickness}_bkg_eval_ttree.root")
sig_eval_tree = sig_eval_file.Get("HitTree")
bkg_eval_tree = bkg_eval_file.Get("HitTree")

# --- OUTPUT --- #
combined_out_file = ROOT.TFile(f"{sensor_thickness}_BDT_combined_training_eval.root", "RECREATE")

def evaluate_flat_tree(flat_tree, scores_list):
    for evt in flat_tree:
        for v in variables:
            buffers[v][0] = getattr(evt, v)
        score = reader.EvaluateMVA("BDT")
        scores_list.append(score)

# --- EVALUATE SIGNAL AND BACKGROUND --- #
combined_out_file.cd()

# --- Training  files --- #
sig_training_scores = []
bkg_training_scores = []
evaluate_flat_tree(sig_training_tree, sig_training_scores)
evaluate_flat_tree(bkg_training_tree, bkg_training_scores)

training_y_true      = np.array([1]*len(sig_training_scores) + [0]*len(bkg_training_scores))
training_y_score     = np.array(sig_training_scores + bkg_training_scores)
training_fpr, training_tpr, training_ = roc_curve(training_y_true, training_y_score)
training_bkg_rej     = 1 - training_fpr
training_sig_eff     = training_tpr
training_roc_auc     = auc(training_sig_eff, training_bkg_rej)
print(f"Training  ROC AUC (Signal efficiency vs Background rejection) = {training_roc_auc:.3f}")

# Save score histograms
h_sig_train = ROOT.TH1F("h_sig_train_score", f"{sensor_thickness} #mum Signal BDT Output (Training);BDT Score;Entries", 100, -1, 1)
for s in sig_training_scores:
    h_sig_train.Fill(s)
h_sig_train_norm = h_sig_train.Clone("h_sig_train_norm")
h_sig_train_norm.Scale(1. / h_sig_train.Integral())

h_bkg_train = ROOT.TH1F("h_bkg_train_score", f"{sensor_thickness} #mum Background BDT Output (Training);BDT Score;Entries", 100, -1, 1)
for b in bkg_training_scores:
    h_bkg_train.Fill(b)
h_bkg_train_norm = h_bkg_train.Clone("h_bkg_train_norm")
h_bkg_train_norm.Scale(1. / h_bkg_train.Integral())

# Non-normalized histograms
c_hist_train = ROOT.TCanvas("c_hist_train", "BDT Output (Non-normalized)", 800, 600)

# Fix y-axis to data max
training_hists      = [h_sig_train, h_bkg_train]
training_global_max = max(h.GetMaximum() for h in training_hists)
training_y_max      = 1.2 * training_global_max

# Draw pretty plot
h_sig_train.SetLineColor(ROOT.kRed)
h_bkg_train.SetLineColor(ROOT.kBlue)
h_sig_train.SetLineWidth(2)
h_bkg_train.SetLineWidth(2)
h_sig_train.SetFillColorAlpha(ROOT.kRed, 0.35)
h_bkg_train.SetFillColorAlpha(ROOT.kBlue, 0.35)
h_sig_train.SetMaximum(training_y_max)
h_sig_train.Draw("HIST")
h_bkg_train.Draw("HIST SAME")
h_sig_train.SetStats(0)

# Add a legend
training_hist_legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
training_hist_legend.AddEntry(h_sig_train, "Signal clusters", "l")
training_hist_legend.AddEntry(h_bkg_train, "Background clusters", "l")
training_hist_legend.SetBorderSize(0)
training_hist_legend.Draw()

c_hist_train.SaveAs(f"{sensor_thickness}_BDT_evaluation_of_training_wpixels.png")
c_hist_train.Write()

# Normalized histograms
c_hist_train_norm = ROOT.TCanvas("c_hist_train_norm", "BDT Output", 800, 600)

# Fix y-axis to data max
training_norm_hists      = [h_sig_train_norm, h_bkg_train_norm]
training_norm_global_max = max(h.GetMaximum() for h in training_norm_hists)
training_norm_y_max      = 1.2 * training_norm_global_max

# Draw pretty plot
h_sig_train_norm.SetLineColor(ROOT.kRed)
h_bkg_train_norm.SetLineColor(ROOT.kBlue)
h_sig_train_norm.SetLineWidth(2)
h_bkg_train_norm.SetLineWidth(2)
h_sig_train_norm.SetFillColorAlpha(ROOT.kRed, 0.35)
h_bkg_train_norm.SetFillColorAlpha(ROOT.kBlue, 0.35)
h_sig_train_norm.SetMaximum(training_norm_y_max)
h_sig_train_norm.Draw("HIST")
h_bkg_train_norm.Draw("HIST SAME")
h_sig_train_norm.SetStats(0)

# Add a legend
training_hist_legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
training_hist_legend.AddEntry(h_sig_train_norm, "Signal clusters", "l")
training_hist_legend.AddEntry(h_bkg_train_norm, "Background clusters", "l")
training_hist_legend.SetBorderSize(0)
training_hist_legend.Draw()

c_hist_train_norm.SaveAs(f"{sensor_thickness}_BDT_evaluation_of_training_wpixels.png")
c_hist_train_norm.Write()

# ROC curve
c_roc_train = ROOT.TCanvas("c_roc_train", "ROC Curve (Signal Eff vs Background Rejection)", 600, 600)
g_train = ROOT.TGraph(len(training_sig_eff), array('f', training_bkg_rej), array('f', training_sig_eff))

training_roc_legend = ROOT.TLegend(0.15, 0.20, 0.35, 0.35)
g_train.SetTitle(f"{sensor_thickness} #mum ROC Curve;Background Rejection;Signal Efficiency")
g_train.SetLineColor(ROOT.kBlue)
g_train.SetLineWidth(2)
g_train.Draw("AL")
g_train.GetXaxis().SetLimits(0, 1)
g_train.GetYaxis().SetRangeUser(0, 1)

c_roc_train.SaveAs(f"{sensor_thickness}_BDT_ROC_of_training_SigEff_vs_BkgRej_wpixels.png")
g_train.Write("training_ROC_curve")

# --- Evaluation files --- #
sig_eval_scores = []
bkg_eval_scores = []
evaluate_flat_tree(sig_eval_tree, sig_eval_scores)
evaluate_flat_tree(bkg_eval_tree, bkg_eval_scores)

eval_y_true      = np.array([1]*len(sig_eval_scores) + [0]*len(bkg_eval_scores))
eval_y_score     = np.array(sig_eval_scores + bkg_eval_scores)
eval_fpr, eval_tpr, eval_ = roc_curve(eval_y_true, eval_y_score)
eval_bkg_rej     = 1 - eval_fpr
eval_sig_eff     = eval_tpr
eval_roc_auc     = auc(eval_sig_eff, eval_bkg_rej)
print(f"Evaluation ROC AUC (Signal efficiency vs Background rejection) = {eval_roc_auc:.3f}")

# Save score histograms
h_sig_eval = ROOT.TH1F("h_sig_eval_score", f"{sensor_thickness} #mum Signal BDT Output (Evaluation);BDT Score;Entries", 100, -1, 1)
for s in sig_eval_scores:
    h_sig_eval.Fill(s)
h_sig_eval_norm = h_sig_eval.Clone("h_sig_eval_norm")
h_sig_eval_norm.Scale(1. / h_sig_eval.Integral())

h_bkg_eval = ROOT.TH1F("h_bkg_eval_score", f"{sensor_thickness} #mum Background BDT Output (Evaluation);BDT Score;Entries", 100, -1, 1)
for b in bkg_eval_scores:
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
eval_hist_legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
eval_hist_legend.AddEntry(h_sig_eval, "Signal clusters", "l")
eval_hist_legend.AddEntry(h_bkg_eval, "Background clusters", "l")
eval_hist_legend.SetBorderSize(0)
eval_hist_legend.Draw()

c_hist_eval.SaveAs(f"{sensor_thickness}_BDT_evaluation_of_eval_wpixels.png")
c_hist_eval.Write()

# Normalized histograms
c_hist_eval_norm = ROOT.TCanvas("c_hist_eval_norm", "BDT Output", 800, 600)

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
eval_hist_legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
eval_hist_legend.AddEntry(h_sig_eval_norm, "Signal clusters", "l")
eval_hist_legend.AddEntry(h_bkg_eval_norm, "Background clusters", "l")
eval_hist_legend.SetBorderSize(0)
eval_hist_legend.Draw()

c_hist_eval_norm.SaveAs(f"{sensor_thickness}_BDT_evaluation_of_eval_wpixels.png")
c_hist_eval_norm.Write()

# ROC curve
c_roc_eval = ROOT.TCanvas("c_roc_eval", "ROC Curve (Signal Eff vs Background Rejection)", 600, 600)
g_eval = ROOT.TGraph(len(eval_sig_eff), array('f', eval_bkg_rej), array('f', eval_sig_eff))

eval_roc_legend = ROOT.TLegend(0.15, 0.20, 0.35, 0.35)
g_eval.SetTitle(f"{sensor_thickness} #mum ROC Curve;Background Rejection;Signal Efficiency")
g_eval.SetLineColor(ROOT.kBlue)
g_eval.SetLineWidth(2)
g_eval.Draw("AL")
g_eval.GetXaxis().SetLimits(0, 1)
g_eval.GetYaxis().SetRangeUser(0, 1)

c_roc_eval.SaveAs(f"{sensor_thickness}_BDT_ROC_of_eval_SigEff_vs_BkgRej_wpixels.png")
g_eval.Write("eval_ROC_curve")

# --- Combined stats --- #
# Save score histograms and ROC
c_hist_comb = ROOT.TCanvas("c_hist_comb", "BDT Output", 800, 600)

# Fix y-axis to data max
norm_hists      = [h_sig_train_norm, h_sig_eval_norm, h_sig_eval_norm, h_bkg_eval_norm]
norm_global_max = max(h.GetMaximum() for h in norm_hists)
norm_y_max      = 1.2 * norm_global_max

h_sig_eval_norm.SetLineColor(ROOT.kRed)
h_bkg_eval_norm.SetLineColor(ROOT.kBlue)
h_sig_train_norm.SetLineColor(ROOT.kMagenta)
h_bkg_train_norm.SetLineColor(ROOT.kCyan)
h_sig_eval_norm.SetMaximum(norm_y_max)
h_sig_eval_norm.Draw("HIST")
h_bkg_eval_norm.Draw("HIST SAME")
h_sig_train_norm.Draw("HIST SAME")
h_bkg_train_norm.Draw("HIST SAME")
h_sig_eval_norm.SetStats(0)

comb_hist_legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
comb_hist_legend.AddEntry(h_sig_eval_norm, "Signal (evaluation data)", "l")
comb_hist_legend.AddEntry(h_sig_train_norm, "Signal (training data)", "l")
comb_hist_legend.AddEntry(h_bkg_eval_norm, "Background (evaluation data)", "l")
comb_hist_legend.AddEntry(h_bkg_train_norm, "Background (training data)", "l")
comb_hist_legend.SetBorderSize(0)
comb_hist_legend.Draw()

c_hist_comb.SaveAs(f"{sensor_thickness}_BDT_evaluation_of_comb_wpixels.png")
c_hist_comb.Write()

# ROC Curve
c_roc_comb = ROOT.TCanvas("c_roc_comb", "ROC Curve (Signal Eff vs Background Rejection)", 600, 600)

comb_roc_legend = ROOT.TLegend(0.15, 0.20, 0.35, 0.35)
g_train.SetLineColorAlpha(ROOT.kBlue,0.5)
g_eval.SetLineColorAlpha(ROOT.kRed,0.5)
g_train.Draw("AL")
g_eval.Draw("L SAME")
comb_roc_legend.AddEntry(g_train,"Training Data","l")
comb_roc_legend.AddEntry(g_eval,"Evaluation Data","l")
comb_roc_legend.SetBorderSize(0)
comb_roc_legend.Draw()

c_roc_comb.SaveAs(f"{sensor_thickness}_BDT_ROC_of_comb_SigEff_vs_BkgRej_wpixels.png")

# Write all to output file
h_sig_eval.Write()
h_bkg_eval.Write()
h_sig_train_norm.Write()
h_bkg_train_norm.Write()

combined_out_file.Close()
