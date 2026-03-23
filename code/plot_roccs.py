import ROOT
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc

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

def get_roc_vals(s_scores, b_scores):
    y_true = np.concatenate([
        np.ones(s_scores.shape[0], dtype=int),
        np.zeros(b_scores.shape[0], dtype=int)
    ])
    y_score     = np.concatenate([s_scores, b_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr

def get_scores(fold_idx=None):
    in_file_tag = kfcv_tag

    if fold_idx is None:
        in_file_suffix = "_eval_score_data.npz"
        in_file_name   = f"{in_file_tag}{in_file_suffix}"
        s_key, b_key   = "eval_sig_scores", "eval_bkg_scores"
    else:
        in_file_suffix = "_cv_score_data.npz"
        in_file_name   = f"{in_file_tag}_fold{fold_idx + 1}{in_file_suffix}"
        s_key, b_key   = "cv_sig_scores", "cv_bkg_scores"

    with np.load(in_file_name) as data:
        return data[s_key], data[b_key]

kfcv_tag = get_kfcv_tag()

# Define ROC curve
out_file = ROOT.TFile(f"{kfcv_tag}_ROC.root", "RECREATE")
out_file.cd()

# Get evaluation data
eval_sig_scores, eval_bkg_scores = get_scores()
eval_fpr, eval_tpr = get_roc_vals(eval_sig_scores, eval_bkg_scores)

eval_rej     = 1 - eval_fpr
eval_eff     = eval_tpr
eval_roc_auc = auc(eval_eff, eval_rej)

print(f"Evaluation ROC AUC: {eval_roc_auc:.3f}")

roc_resolution = len(eval_eff)
fpr_grid       = np.linspace(0, 1, roc_resolution)

# Get CrossValidation data
test_tpr_folds = []

for fold in range(5):
    test_sig_scores, test_bkg_scores = get_scores(fold)
    test_fpr, test_tpr = get_roc_vals(test_sig_scores, test_bkg_scores)
    # Interpolate TPR onto common FPR grid
    test_tpr_interp = np.interp(fpr_grid, test_fpr, test_tpr)
    test_tpr_folds.append(test_tpr_interp)

test_tpr_folds = np.array(test_tpr_folds) # shape (5, roc_resolution)

test_fpr_folds = []

for fold in range(5):
    test_sig_scores, test_bkg_scores = get_scores(fold)
    test_fpr, test_tpr = get_roc_vals(test_sig_scores, test_bkg_scores)
    # Interpolate FPR onto common TPR grid
    test_fpr_interp = np.interp(fpr_grid, test_tpr, test_fpr)
    test_fpr_folds.append(test_fpr_interp)

test_fpr_folds = np.array(test_fpr_folds) # shape (5, roc_resolution)

test_mean_tpr  = np.mean(test_tpr_folds, axis=0)
test_std_tpr   = np.std(test_tpr_folds, axis=0)
test_mean_rej  = 1 - fpr_grid
test_mean_eff  = test_mean_tpr
test_std_rej   = np.std(test_fpr_folds, axis=0)
test_roc_auc   = auc(test_mean_eff, test_mean_rej)

print(f"CrossValidation test mean ROC AUC: {test_roc_auc:.3f}")

# Plot test uncertainty band
g_vert_band = ROOT.TGraph(2*roc_resolution)

for i in range(roc_resolution): # upper edge, vertical
    g_vert_band.SetPoint(
        i,
        eval_eff[i],
        eval_rej[i] + test_std_rej[i]
    )

for i in range(roc_resolution): # lower edge, vertical (reverse order to ensure shape closure)
    g_vert_band.SetPoint(
        roc_resolution + i,
        eval_eff[roc_resolution - 1 - i],
        eval_rej[roc_resolution - 1 - i] - test_std_rej[roc_resolution - 1 - i]
    )

g_vert_band.SetFillColorAlpha(ROOT.kBlue, 0.3)
g_vert_band.SetLineColor(0)

g_horz_band = ROOT.TGraph(2*roc_resolution)

for i in range(roc_resolution): # upper edge, horizontal
    g_horz_band.SetPoint(
        i,
        eval_eff[i] + test_std_tpr[i],
        eval_rej[i]
    )

for i in range(roc_resolution): # lower edge, horizontal (reverse order to ensure shape closure)
    g_horz_band.SetPoint(
        roc_resolution + i,
        eval_eff[roc_resolution - 1 - i] - test_std_tpr[roc_resolution - 1 - i],
        eval_rej[roc_resolution - 1 - i]
    )

g_horz_band.SetFillColorAlpha(ROOT.kBlue, 0.3)
g_horz_band.SetLineColor(0)

# Plot evaluation ROC curve
eval_n = len(eval_eff)

g_eval = ROOT.TGraph(eval_n)

for i in range(eval_n):
    g_eval.SetPoint(i, eval_eff[i], eval_rej[i])

g_eval.SetLineColor(ROOT.kRed)
g_eval.SetLineWidth(2)

# Draw stuff
c = ROOT.TCanvas(f"roc_{kfcv_tag}", f"BDT ROC Curve {kfcv_tag}", 800, 600)
c.cd()

g_vert_band.SetTitle(f"BDT ROC Curve {kfcv_tag}")

g_vert_band.Draw("AF")
g_horz_band.Draw("AF")
g_eval.Draw("L SAME")

g_vert_band.GetXaxis().SetTitle("Signal efficiency")
g_vert_band.GetYaxis().SetTitle("Background rejection")

legend = ROOT.TLegend(0.12, 0.12, 0.35, 0.25)
legend.AddEntry(g_eval, "Held-out evaluation sample", "l")
legend.AddEntry(g_vert_band, "#pm1#sigma TPR at fixed FPR across folds", "f")
legend.SetBorderSize(0)
legend.Draw()

c.Modified()
c.Update()

c.SaveAs(f"{kfcv_tag}_rocc.pdf")

out_file.cd()

c.Write()
g_vert_band.Write()
g_horz_band.Write()
g_eval.Write()

out_file.Close()
