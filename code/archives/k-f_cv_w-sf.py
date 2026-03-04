"""
K-fold cross validation with successive fractioning.
TODO : Insert explanation of wtf that all is here.
"""
import os
import csv
import ROOT
import random
from ROOT import TMVA
from dataclasses import dataclass

"""
Train/test split is managed by the number of folds (k_folds), with
training = k-1 folds of clusters
test = 1 fold of clusters
"""

# Parameters
k_folds        = 5
n_trees        = 400
min_node_size  = ["1.5%", "2.0%", "2.5%", "3%", "3.5%"]
max_depth      = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
beta           = .1

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

def aucs_to_csv(csv_path, *, thickness, node_size, depth, ntrees, b, fold_rows, auc_avg, auc_std):
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
                "n_trees": ntrees,
                "min_node_size": node_size,
                "max_depth": depth,
                "beta": b,
                "fold": r["fold"],
                "auc": r["auc"],
                "auc_avg": auc_avg,
                "auc_std": auc_std
            })

@dataclass(frozen=True)
class BDTConfig:
    node_size: str
    depth: int
    beta: float

def make_initial_pool(*, pool_size=60, seed=123):
    """
    Random search for configs (better than full grid for high-dim).
    You can increase pool_size based on compute budget.
    """
    rng = random.Random(seed)
    pool = []
    for _ in range(pool_size):
        cfg = BDTConfig(
            node_size=rng.choice(min_node_size),
            depth=rng.choice(max_depth),
            beta=rng.choice(beta),
        )
        pool.append(cfg)
    seen = set()
    uniq = []
    for c in pool:
        key = (c.node_size, c.depth, c.beta)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def successive_fractioning(thickness, *, n_candidates=60, budgets=None, eta=3, seed=123):
    """
    budgets: increasing NTrees values (budget levels)
    eta: keep fraction = 1/eta each round
    """
    if budgets is None:
        budgets = [200, 400, 800, 1200]  

    pool = make_initial_pool(pool_size=n_candidates, seed=seed)

    for round_idx, ntrees_budget in enumerate(budgets):
        scored = []
        stage_tag = f"_stage{round_idx}_N{ntrees_budget}"

        for cfg in pool:
            auc_avg = run_tmva_kfold(
                thickness,
                cfg.node_size,
                cfg.depth,
                ntrees_budget,
                cfg.beta,
                stage_tag=stage_tag,
            )
            scored.append((auc_avg, cfg))

        # Rank by AUC descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top fraction, but never drop below 1
        keep = max(1, len(scored) // eta)
        pool = [cfg for _, cfg in scored[:keep]]

        if len(pool) == 1:
            break

    best_auc, best_cfg = scored[0]
    return best_cfg, best_auc

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

def run_tmva_kfold(thickness, node_size, depth, ntrees, b, *, stage_tag=""):
    TMVA.Tools.Instance()

    out_file_name = f"{thickness}{stage_tag}_TMVACV.root"
    output_file = ROOT.TFile.Open(out_file_name, "RECREATE")

    dataloader = TMVA.DataLoader("datasetcv")
    for v_id in variables.keys():
        dataloader.AddVariable(v_id, "F")

    sig_file = ROOT.TFile(f"/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/data/MAIA/signal/{thickness}_sig_trng_ttree_k{k_folds}.root")
    bkg_file = ROOT.TFile(f"/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/data/MAIA/bg/{thickness}_bkg_trng_ttree_k{k_folds}.root")
    sig_tree = sig_file.Get("HitTree")
    bkg_tree = bkg_file.Get("HitTree")

    dataloader.AddSignalTree(sig_tree, 1.0)
    dataloader.AddBackgroundTree(bkg_tree, 1.0)

    dataloader.PrepareTrainingAndTestTree(
        "", "",
        "nTrain_Signal=0:nTrain_Background=0:"
        "nTest_Signal=0:nTest_Background=0:!V"
    )
    dataloader.AddSpectator("fold", "I")

    cv_opts = (
        f"!V:"
        f"AnalysisType=Classification:"
        f"ROC:"
        f"NumFolds={k_folds}:"
        f"SplitType=Deterministic:"
        f"SplitExpr=[fold]"
    )

    cv = TMVA.CrossValidation("cv_job", dataloader, output_file, cv_opts)

    bookmethod_opts = (
        f"!H:!V:"
        f"NTrees={ntrees}:MaxDepth={depth}:MinNodeSize={node_size}:"
        f"BoostType=AdaBoost:AdaBoostBeta={b}:"
        f"SeparationType=GiniIndex:nCuts=20"
    )

    cv.BookMethod(TMVA.Types.kBDT, "BDT", bookmethod_opts)
    cv.Evaluate()

    results = cv.GetResults()
    fold_rows, auc_avg, auc_std = extract_aucs(results)

    aucs_to_csv(
        results_csv_path,
        thickness=thickness,
        node_size=node_size,
        depth=depth,
        ntrees=ntrees,
        b=b,
        fold_rows=fold_rows,
        auc_avg=auc_avg,
        auc_std=auc_std,
    )

    output_file.Close()
    return auc_avg

sensor_thicknesses = [50]

for t in sensor_thicknesses:
    best_cfg, best_auc = successive_fractioning(
        t,
        n_candidates=5733,
        budgets=[800, 1000, 1200, 1400],
        eta=3,
        seed=123,
    )
    print(f"[thickness={t}] best AUC={best_auc:.4f} with {best_cfg}")
