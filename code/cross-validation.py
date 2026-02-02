import ROOT
from ROOT import TMVA
from dataclasses import dataclass

"""
Train/test split is managed by the number of folds (k), with
training = k-1 folds of clusters
test = 1 fold of clusters
"""
k_folds        = 5
n_trees        = 800
clust_per_tree = "3%"
max_depth      = 8
beta           = .5

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

    # Cross validation controller
    cv = TMVA.CrossValidation("cv_job", dataloader, output_file, f"!V:NumFolds={k_folds}")
    # Allow comparison across hyperparameter tuning outputs
    # Determines fold split by modulus of TTree entry index
    # with number of folds (k_folds), cycling from 0 to k-1
    # NOTE: this will not hold if TTrees are altered later!
    cv.SetSplitExpr("Entry$ % int([NumFolds])")

    # Book method(s) as with factory
    bookmethod_opts = (
        f"!H:!V:NTrees={n_trees}:MaxDepth={max_depth}:"
        f"MinNodeSize={clust_per_tree}:BoostType=AdaBoost:"
        f"AdaBoostBeta={beta}:SeparationType=GiniIndex:nCuts=20"
    )

    cv.BookMethod(
        TMVA.Types.kBDT,
        "BDT", bookmethod_opts
    )

    # Train, test, and evaluate booked method(s)
    cv.Evaluate()

    # Fold-by-fold results
    results = cv.GetResults()  # vector<CrossValidationResult>
    print(f"Cross validation completed. Results saved to {out_file_name}.")
    print("------------------------")
    print(f"-------{thickness} micron-------")
    print("------------------------")
    print("CROSS VALIDATION SUMMARY")
    print(results)

    output_file.Close()

sensor_thicknesses = [50, 75, 100, 200, 400]

for t in sensor_thicknesses:
    run_tmva_kfold(t)
