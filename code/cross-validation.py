import os
import csv
import ROOT
from ROOT import TMVA
from dataclasses import dataclass

"""
Train/test split is managed by the number of folds (k_folds), with
training = k-1 folds of clusters
test = 1 fold of clusters
"""

# Parameters
k_folds        = 5
n_trees        = 800
min_node_size  = "3%"
max_depth      = 8
beta           = .5

# I/O
results_csv_name = "cv_aucs.csv"
results_csv_path = "/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/code/" + results_csv_name 

# Pixel information helper function
def make_pixelhit_vars(prefix, label, *, n=9, ymax, xmin, xmax, legend="right", yscale="log"):
    # Sorry, Tova. Dictionary comprehension just looks so much better here.
    return {
        f"{prefix}_{i}": Variable(
            label=f"{label} for pixel hit {i}",
            ymax=ymax,
            xmin=xmin,
            xmax=xmax,
            legend=legend,
            yscale=yscale,
        )
        for i in range(n)
    }

def extract_aucs(cv_results):
    """
    Takes cv.GetResults() object.
    Returns a list of dictionary entries:
        [{"fold": 0, "auc": 0.918}, ...]
    as well as the average and standard 
    deviation results.
    """

    # Error checking
    if len(cv_results) <= 0:
        raise RuntimeError(f"Cross validation returned {len(cv_results)} results.")
    
    # Process first booked method
    r0       = cv_results[0] # focus on first booked method
    roc_map0 = r0.GetROCValues() # std::map<UInt_t, Float_t> 
    
    # Iterate over ROC map
    rows0 = []
    i     = roc_map0.begin()
    end   = roc_map0.end()
    while i != end:
        pair = i.__deref__() # get fold/AUC pairs from map
        rows0.append({
            "fold": int(pair.first),
            "auc": float(pair.second),
        })
        i.__preinc__() # preincrement iterator (++i)
    
    rows0.sort(key=lambda d: d["fold"]) # sort rows by fold index
    auc_avg0 = float(r0.GetROCAverage())
    auc_std0 = float(r0.GetROCStandardDeviation())
    
    # TODO: Expand for future use of multiple methods
    return rows0, auc_avg0, auc_std0

def aucs_to_csv(csv_path, *, thickness, fold_rows, auc_avg, auc_std):
    fields = [
        "thickness", "k", "n_trees", "min_node_size", "max_depth",
        "beta", "fold", "auc", "auc_avg", "auc_std"
    ]
    # Check for existing .csv, write if none
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        for r in fold_rows:
            w.writerow({
                "thickness": thickness,
                "k": k_folds,
                "n_trees": n_trees,
                "min_node_size": min_node_size,
                "max_depth": max_depth,
                "beta": beta,
                "fold": r["fold"],
                "auc": r["auc"],
                "auc_avg": auc_avg,
                "auc_std": auc_std
            })

# Create variable class
@dataclass(frozen=True)
class Variable:
    label: str  # x-axis label for histograms
    ymax: float # max y-value on histograms
    xmin: float # min x-value on histograms
    xmax: float # max x-value on histograms
    legend: str # legend position tag
    yscale: str # linear or log scale for y-axis

# Create cluster variable objects
variables = {
        "Cluster_ArrivalTime": Variable(
            label="Cluster arrival time [ns]",
            ymax=0.3,
            xmin=-0.52,
            xmax=0.7,
            legend="right",
            yscale="linear",
        ),
        "Cluster_EnergyDeposited": Variable(
            label="Cluster energy deposited [KeV]",
            ymax=1.1,
            xmin=0,
            xmax=0.0007,
            legend="right",
            yscale="log",
        ),
        "Incident_Angle": Variable(
            label="Incident angle [radians]",
            ymax=0.11,
            xmin=0,
            xmax=3,
            legend="center",
            yscale="linear",
        ),
        "Cluster_Size_x": Variable(
            label="Cluster size in x [pixels]",
            ymax=1.1,
            xmin=0,
            xmax=60,
            legend="right",
            yscale="log",
        ),
        "Cluster_Size_y": Variable(
            label="Cluster size in y [pixels]",
            ymax=1.1,
            xmin=0,
            xmax=400,
            legend="right",
            yscale="log",
        ),
        "Cluster_Size_tot": Variable(
            label="Total cluster size [pixels]",
            ymax=1.1,
            xmin=0,
            xmax=400,
            legend="right",
            yscale="log",
        ),
        "Cluster_x": Variable(
            label="Cluster x position [cm]",
            ymax=0.11,
            xmin=-35,
            xmax=35,
            legend="center",
            yscale="linear",
        ),
        "Cluster_y": Variable(
            label="Cluster y position [cm]",
            ymax=0.11,
            xmin=-35,
            xmax=35,
            legend="center",
            yscale="linear",
        ),
        "Cluster_z": Variable(
            label="Cluster z position [cm]",
            ymax=1.1,
            xmin=-80,
            xmax=80,
            legend="right",
            yscale="log",
        ),
        "Cluster_RMS_x": Variable(
            label="Cluster RMS in x [cm^{2}]",
            ymax=0.22,
            xmin=0,
            xmax=90000,
            legend="right",
            yscale="linear",
        ),
        "Cluster_RMS_y": Variable(
            label="Cluster RMS in y [cm^{2}]",
            ymax=0.22,
            xmin=0,
            xmax=350000,
            legend="right",
            yscale="linear",
        ),
        "Cluster_Skew_x": Variable(
            label="Cluster skew in x",
            ymax=0.55,
            xmin=-1.75,
            xmax=1.75,
            legend="center",
            yscale="log",
        ),
        "Cluster_Skew_y": Variable(
            label="Cluster skew in y",
            ymax=0.55,
            xmin=-1.75,
            xmax=1.75,
            legend="center",
            yscale="log",
        ),
        "Cluster_AspectRatio": Variable(
            label="Cluster aspect ratio",
            ymax=1.1,
            xmin=0,
            xmax=25000,
            legend="right",
            yscale="log",
        ),
        }

# Create pixel variable objects
variables |= make_pixelhit_vars(
    "PixelHits_EnergyDeposited",
    "Pixel hit energy deposited [KeV]",
    ymax=13250,
    xmin=0,
    xmax=62000,
)

variables |= make_pixelhit_vars(
    "PixelHits_ArrivalTime",
    "Pixel hit arrival time [ns]",
    ymax=13500,
    xmin=-2.1,
    xmax=5.2,
)

# --- K-FOLD CROSS VALIDATION --- #
out_file_suffix = "_TMVACV.root"

def run_tmva_kfold(thickness):
    TMVA.Tools.Instance()
    # Create output file
    out_file_name = str(thickness) + out_file_suffix
    output_file = ROOT.TFile.Open(out_file_name, "RECREATE")

    dataloader = TMVA.DataLoader("datasetcv")
    # Load input variables
    for v_id, _ in variables.items():
        dataloader.AddVariable(v_id, "F")

    # Load signal and background files
    sig_file = ROOT.TFile(f"/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/data/MAIA/signal/{thickness}_sig_trng_ttree_k{k_folds}.root")
    bkg_file = ROOT.TFile(f"/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/data/MAIA/bg/{thickness}_bkg_trng_ttree_k{k_folds}.root")
    sig_tree = sig_file.Get("HitTree")
    bkg_tree = bkg_file.Get("HitTree")

    dataloader.AddSignalTree(sig_tree, 1.0)
    dataloader.AddBackgroundTree(bkg_tree, 1.0)

    # Prepare dataset
    dataloader.PrepareTrainingAndTestTree(
        "", "",
        "nTrain_Signal=0:nTrain_Background=0:"
        "nTest_Signal=0:nTest_Background=0:!V"
    )
    dataloader.AddSpectator("fold", "I")
    
    cv_opts = (
        f"!V:"
        f"AnalysisType=Classification:"
        f"ROC:"                          # enable ROC filling into CrossValidationResult
        f"NumFolds={k_folds}:"
        f"SplitType=Deterministic:"
        f"SplitExpr=[fold]"              # allow comparison across hyperparameter tuning outputs
    )

    # Cross validation controller
    cv = TMVA.CrossValidation("cv_job", dataloader, output_file, cv_opts)

    # Book method(s) as with factory
    bookmethod_opts = (
        f"!H:!V:NTrees={n_trees}:MaxDepth={max_depth}:"
        f"MinNodeSize={min_node_size}:BoostType=AdaBoost:"
        f"AdaBoostBeta={beta}:SeparationType=GiniIndex:nCuts=20"
    )

    cv.BookMethod(
        TMVA.Types.kBDT,
        "BDT", bookmethod_opts
    )

    # Train, test, and evaluate booked method(s)
    cv.Evaluate()

    # Get and store fold-by-fold results
    results = cv.GetResults()
    print("Sanity check print")
    results[0].Print()

    fold_rows, auc_avg, auc_std = extract_aucs(results)
    aucs_to_csv(
        results_csv_path,
        thickness=thickness,
        fold_rows=fold_rows,
        auc_avg=auc_avg,
        auc_std=auc_std,
    )

    print(f"Cross validation completed. Results saved to {out_file_name} and written to {results_csv_name}.")
    print("------------------------")
    print(f"-------{thickness} micron-------")
    print("------------------------")
    print("CROSS VALIDATION SUMMARY")
    print("AUCs:", fold_rows)
    print("avg/std:", auc_avg, auc_std)
    print("Split expr:", cv.GetSplitExpr())
    print("Num folds :", cv.GetNumFolds())

    output_file.Close()

#sensor_thicknesses = [50, 75, 100, 200, 400]
sensor_thicknesses = [400]

for t in sensor_thicknesses:
    run_tmva_kfold(t)
