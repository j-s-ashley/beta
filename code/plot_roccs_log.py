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

roc_resolution = 500
fpr_grid       = np.linspace(0, 1, roc_resolution)

# Get CrossValidation data
test_tpr_folds = []

for fold in range(5):
    test_sig_scores, test_bkg_scores = get_scores(fold)
    test_fpr, test_tpr = get_roc_vals(test_sig_scores, test_bkg_scores)
    # Interpolate TPR onto common FPR grid
    test_tpr_interp = np.interp(fpr_grid, test_fpr, test_tpr)
    test_tpr_folds.append(test_tpr_interp)

test_tpr_folds = np.array(test_tpr_folds)          # shape (5, roc_resolution)
test_mean_tpr  = np.mean(test_tpr_folds, axis=0)
test_std_tpr   = np.std(test_tpr_folds, axis=0)
test_mean_rej  = 1 - fpr_grid
test_mean_eff  = test_mean_tpr
test_std_rej   = test_std_tpr                      # rejection uses same var
test_roc_auc   = auc(test_mean_eff, test_mean_rej)

print(f"CrossValidation test mean ROC AUC: {test_roc_auc:.3f}")

# Get evaluation data
eval_sig_scores, eval_bkg_scores = get_scores()
eval_fpr, eval_tpr = get_roc_vals(eval_sig_scores, eval_bkg_scores)

eval_rej     = 1 - eval_fpr
eval_eff     = eval_tpr
eval_roc_auc = auc(eval_eff, eval_rej)

print(f"Evaluation ROC AUC: {eval_roc_auc:.3f}")

# Plot mean test ROC curve
test_n = len(test_mean_eff)

g_test_mean = ROOT.TGraph(test_n)

# Safety clipping against log10(0) break
epsilon       = 1e-12 # min allowed value
test_mean_rej = np.clip(test_mean_rej, epsilon, 1)
eval_rej      = np.clip(eval_rej, epsilon, 1)

for i in range(test_n):
    g_test_mean.SetPoint(i, test_mean_eff[i], 1 - np.log10(test_mean_rej[i]))

g_test_mean.SetLineWidth(2)
g_test_mean.SetLineColor(ROOT.kBlue)

# Plot test uncertainty band
g_test_band = ROOT.TGraph(2*test_n)

for i in range(test_n): # upper edge
    y_upper = 1 - np.log10(test_mean_rej[i] + test_std_rej[i])
    g_test_band.SetPoint(
        i,
        test_mean_eff[i],
        y_upper
    )

for i in range(test_n): # lower edge (reverse order to ensure shape closure)
    y_lower = 1 - np.log10(test_mean_rej[i] - test_std_rej[i])
    g_test_band.SetPoint(
        test_n + i,
        test_mean_eff[test_n - 1 - i],
        y_lower
    )

g_test_band.SetFillColorAlpha(ROOT.kBlue, 0.3)
g_test_band.SetLineColor(0)

# Plot evaluation ROC curve
eval_n = len(eval_eff)

g_eval = ROOT.TGraph(eval_n)

for i in range(eval_n):
    g_eval.SetPoint(i, eval_eff[i], 1 - np.log10(eval_rej[i]))

g_eval.SetLineColor(ROOT.kRed)
g_eval.SetLineWidth(2)

# Draw stuff
c = ROOT.TCanvas(f"roc_{kfcv_tag}", f"BDT ROC Curve {kfcv_tag}", 800, 600)
c.cd()

g_test_band.SetTitle(f"BDT ROC Curve {kfcv_tag}")

g_test_band.Draw("AF")
g_test_mean.Draw("L SAME")
g_eval.Draw("L SAME")

g_test_band.GetXaxis().SetTitle("Signal efficiency")
g_test_band.GetYaxis().SetTitle("1 - log_{10}(Background rejection)")
g_test_band.SetMinimum(0)
g_test_band.SetMaximum(1)

legend = ROOT.TLegend(0.12, 0.75, 0.35, 0.88)
legend.AddEntry(g_test_mean, "Mean k-fold CV", "l")
legend.AddEntry(g_eval, "Held-out evaluation sample", "l")
legend.AddEntry(g_test_band, "#pm1#sigma TPR at fixed FPR across folds", "f")
legend.SetBorderSize(0)
legend.Draw()

c.SetLogy()

c.Modified()
c.Update()

c.SaveAs(f"{kfcv_tag}_rocc_log_rej.pdf")

out_file.cd()

c.Write()
g_test_band.Write()
g_test_mean.Write()
g_eval.Write()

out_file.Close()
