import ROOT
import numpy as np

# Open ROOT file and retreive histograms
f = ROOT.TFile("BDT_combined_training_eval.root")
h_sig_score = f.Get("h_sig_eval_score")
h_bkg_score = f.Get("h_bkg_eval_score")

# Create and open output file
out_file = ROOT.TFile("fraction_per_score.root", "RECREATE")
out_file.cd()

# Get total number of clusters
tot_sig_clusters = h_sig_score.Integral(0, h_sig_score.GetNbinsX()+1)
tot_bkg_clusters = h_bkg_score.Integral(0, h_bkg_score.GetNbinsX()+1)

def get_clusters_per_score(hist):
    """
    This function tracks the cumulative total of clusters remaining
    based on a BDT score cut.

    Args:
        hist (TH1F): Root histogram of clusters, binned per BDT score.

    Returns:
        NumPy arrays of clusters and scores.
    """
    scores       = []
    clusters     = []
    cum_clusters = 0
    
    for i in range(hist.GetNbinsX(), 0, -1):
        scores.append(hist.GetBinCenter(i))
        cum_clusters += hist.GetBinContent(i)
        clusters.append(cum_clusters)

    return np.array(clusters, dtype=float), np.array(scores, dtype=float)

sig_clusters, sig_scores = get_clusters_per_score(h_sig_score)
bkg_clusters, bkg_scores = get_clusters_per_score(h_bkg_score)

# Get cumulative fractions of remaining clusters
sig_fractions = sig_clusters / tot_sig_clusters
bkg_fractions = bkg_clusters / tot_bkg_clusters

c     = ROOT.TCanvas("cluster_frac_per_score", "Cumulative Remaining Cluster Fraction Per BDT Score Cut", 800, 600)
g_sig = ROOT.TGraph(len(sig_scores), sig_scores, sig_fractions)
g_bkg = ROOT.TGraph(len(bkg_scores), bkg_scores, bkg_fractions)

legend = ROOT.TLegend(0.15, 0.20, 0.35, 0.35)
g_sig.SetLineColor(ROOT.kRed)
g_bkg.SetLineColor(ROOT.kBlue)
g_sig.SetTitle("Cumulative Remaining Cluster Fraction Per BDT Score Cut;BDT Score; Remaining Cluster Fraction")
g_sig.Draw("AL")
g_bkg.Draw("L SAME")
legend.AddEntry(g_sig,"Signal","l")
legend.AddEntry(g_bkg,"Background","l")
legend.Draw()

c.SaveAs("cluster_frac_per_score.pdf")

c.Write()
h_sig_score.Write()
h_bkg_score.Write()

out_file.Close()
