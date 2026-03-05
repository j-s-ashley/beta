import glob
import ROOT
import argparse
import numpy as np

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

kfcv_tag = (
    str(sensor_thickness) + "_" +
    str(min_node_size[0]) + "-" +
    str(min_node_size[2]) + "_" +
    str(max_depth) + "_" +
    str(n_trees) + "_" +
    str(beta[0]) + "-" +
    str(beta[2])
)
out_file_suffix = "_cv_score_data.npz"

def export_cv_scores_to_npz(root_basename, method_name="BDT"):

    fold_files = sorted(glob.glob(f"{root_basename}_fold*.root"))

    if len(fold_files) == 0:
        raise RuntimeError("No fold files found. Check FoldFileOutput=True.")

    for fold_idx, fname in enumerate(fold_files):

        sig_scores = []
        bkg_scores = []

        f = ROOT.TFile.Open(fname)
        tree = f.Get("datasetcv/TestTree")

        if not tree:
            raise RuntimeError(f"No TestTree in {fname}")

        branch_name = f"{method_name}_fold{fold_idx+1}"
        for event in tree:
            score = getattr(event, branch_name)   # e.g. event.BDT
            if event.classID == 1:
                bkg_scores.append(score)
            else:
                sig_scores.append(score)

        f.Close()

        # Convert to numpy
        cv_sig_scores = np.array(sig_scores)
        cv_bkg_scores = np.array(bkg_scores)

        out_file_name = f"{kfcv_tag}_fold{fold_idx+1}{out_file_suffix}"
        np.savez(out_file_name,
             cv_sig_scores=cv_sig_scores,
             cv_bkg_scores=cv_bkg_scores
        )

        print(f"Saved {len(cv_sig_scores)} signal and "
          f"{len(cv_bkg_scores)} background scores to {out_file_name}"
        )

export_cv_scores_to_npz("BDT", method_name="BDT")
