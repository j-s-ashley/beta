import os
import ROOT
"""
Add "fold" TTree branches. Allows deterministic split
(same clusters in the same folds) in cross validation
and, thereby, comparison across hyperparameter tuning
outputs. Fold deteremined by TTree entry index mod k,
cycling from 0 to k-1. Appends k val to output files.
NOTE: This will not hold if TTrees are altered later!
"""

k = 5

def write_folded_file(in_path, out_path, tree_name="HitTree"):
    df = ROOT.RDataFrame(tree_name, in_path)
    df2 = df.Define("fold", f"int(rdfentry_ % {k})")
    # Snapshot keeps all branches + adds fold
    df2.Snapshot(tree_name, out_path)

def folded_path(path):
    base, ext = os.path.splitext(path)
    return f"{base}_k{k}{ext}"

sensor_thickness = [50, 75, 100, 200, 400]

for thickness in sensor_thickness:
    sig_in = f"/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/data/MAIA/signal/{thickness}_sig_trng_ttree.root"
    bkg_in = f"/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/data/MAIA/bg/{thickness}_bkg_trng_ttree.root"

    sig_out = folded_path(sig_in)
    bkg_out = folded_path(bkg_in)

    write_folded_file(sig_in, sig_out, "HitTree")
    write_folded_file(bkg_in, bkg_out, "HitTree")

    print(f"Folded TTree for {thickness} micron signal data written to")
    print(sig_out)
    print(f"Folded TTree for {thickness} micron background data written to")
    print(bkg_out)
