import ROOT
import numpy as np
import matplotlib.pyplot as plt
from ROOT import TMVA
from array import array

def train_bdt(n_trees, sig_tree, bkg_tree, variables):
    # Create output file
    output_file = ROOT.TFile("TMVA_output.root", "RECREATE")

    # Initialize TMVA
    TMVA.Tools.Instance()
    factory = TMVA.Factory("TMVAClassification", output_file,
        "!V:!Silent:Color:DrawProgressBar:Transformations=I;:AnalysisType=Classification")
    dataloader = TMVA.DataLoader("dataset")

    for v in variables:
        dataloader.AddVariable(v, "F")

    for i in range(9):
        dataloader.AddVariable(f"PixelHits_EnergyDeposited_{i}", "F")
        dataloader.AddVariable(f"PixelHits_ArrivalTime_{i}", "F")

    dataloader.AddSignalTree(sig_tree, 1.0)
    dataloader.AddBackgroundTree(bkg_tree, 1.0)

    # Prepare dataset
    dataloader.PrepareTrainingAndTestTree(ROOT.TCut(""), ROOT.TCut(""),
        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V")

    # Book a BDT
    factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDT",
            "!H:!V:NTrees=200:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20")

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    output_file.Close()

def evaluate_accuracy(weightfile, test_tree_name="dataset/TestTree"):
    reader = ROOT.TMVA.Reader()

    pixel_vars = []
    for i in range(9):
        pixel_vars.append(f"PixelHits_EnergyDeposited_{i}")
        pixel_vars.append(f"PixelHits_ArrivalTime_{i}")

    reader_vars = variables + pixel_vars

    var_arrays = {}
    for v in reader_vars:
        var_arrays[v] = array('f', [0.0])
        reader.AddVariable(v, var_arrays[v])

    reader.BookMVA("BDT", weightfile)

    # load test tree
    f = ROOT.TFile.Open("TMVA_output.root")
    test_tree = f.Get(test_tree_name)
    if not test_tree:
        raise RuntimeError(f"Could not find tree '{test_tree_name}'")

    correct = 0
    total = 0

    for event in test_tree:
        for v in reader_vars:
            var_arrays[v][0] = getattr(event, v)

        score = reader.EvaluateMVA("BDT")
        truth = int(event.classID)  # 0 = bkg, 1 = sig
        pred = 1 if score > 0 else 0   # simple threshold

        correct += int(pred == truth)
        total += 1

    f.Close()

    if total == 0:
        raise RuntimeError("No events found in test tree")

    return correct / total

def plot_epoch_accuracy(accuracies, savepath=None):
    """
    Plot accuracy vs. epoch for a TMVA BDT.

    Parameters
    ----------
    accuracies : list or array-like
        Accuracy values in [0,1] for each epoch (index = epoch-1)
    savepath : str or None
        If given, saves to this path; otherwise just shows the plot.
    """

    epochs = np.arange(1, len(accuracies) + 1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, accuracies, marker='o', linewidth=1.4)

    plt.title("BDT Accuracy per Epoch")
    plt.xlabel("Epoch (Number of Boosting Rounds / NTrees)")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    else:
        plt.show()

# Define input variables
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
    "Cluster_Eccentricity"
]

# Load signal and background files
sig_file = ROOT.TFile("/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/signal/Hits_TTree_output_digi_light_training.root")
bkg_file = ROOT.TFile("/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/bg/Hits_TTree_output_digi_light_training.root")
sig_tree = sig_file.Get("HitTree")
bkg_tree = bkg_file.Get("HitTree")

accuracies = []

for n in range(1, 201):   # 200 epochs
    train_bdt(n, sig_tree, bkg_tree, variables)

    weightfile = f"dataset/weights/TMVAClassification_BDT.weights.xml"
    acc = evaluate_accuracy(weightfile)
    accuracies.append(acc)

plot_epoch_accuracy(accuracies, savepath="bdt_accuracy_curve.png")
