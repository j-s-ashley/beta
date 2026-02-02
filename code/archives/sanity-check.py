import ROOT
from ROOT import TMVA
from dataclasses import dataclass

"""
Train/test split is managed by the number of folds (k), with
training = k-1 folds of clusters
test = 1 fold of clusters
"""

k_folds = 5
ts      = [50, 75, 100, 200, 400]

for t in ts:
    TMVA.Tools.Instance()
    sig_file = ROOT.TFile(f"/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/data/MAIA/signal/{t}_sig_trng_ttree_k{k_folds}.root")
    bkg_file = ROOT.TFile(f"/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/data/MAIA/bg/{t}_bkg_trng_ttree_k{k_folds}.root")
    sig_tree = sig_file.Get("HitTree")
    bkg_tree = bkg_file.Get("HitTree")
    print("sig branches:", [b.GetName() for b in sig_tree.GetListOfBranches()])
    print("bkg branches:", [b.GetName() for b in bkg_tree.GetListOfBranches()])
