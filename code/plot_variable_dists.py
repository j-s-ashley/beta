import ROOT
import math
import argparse
from dataclasses import dataclass

def get_ttree_clusters(t):
    n = int(t.GetEntriesFast())
    return n

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

# Manual normalization
def normalize_in_place(h):
    integ = h.Integral()
    if integ > 0:
        h.Scale(1.0 / integ)

@dataclass(frozen=True)
class Variable:
    label: str  # x-axis label for histograms
    ymax: float # max y-value on histograms
    xmin: float # min x-value on histograms
    xmax: float # max x-value on histograms
    legend: str # legend position tag
    yscale: str # linear or log scale for y-axis

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

# --- INPUT VARIABLE DISTRIBUTIONS --- #
input_path_stub    = "/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/data/MAIA"
datasets           = ["trng", "eval"]
sensor_thicknesses = [50, 75, 100, 200, 400]
n_bins             = 50
ROOT.gStyle.SetOptStat(0)

for sensor_thickness in sensor_thicknesses:
    for dataset in datasets:
        out_file = ROOT.TFile(f"{sensor_thickness}_{dataset}.root", "RECREATE")
        if not out_file or out_file.IsZombie():
            raise RuntimeError("Could not create output file")

        sig_file = ROOT.TFile(f"{input_path_stub}/signal/{sensor_thickness}_sig_{dataset}_ttree.root", "READ")
        bkg_file = ROOT.TFile(f"{input_path_stub}/bg/{sensor_thickness}_bkg_{dataset}_ttree.root", "READ")
        if sig_file.IsZombie() or bkg_file.IsZombie():
            raise RuntimeError("Bad input file")

        sig_tree = sig_file.Get("HitTree")
        bkg_tree = bkg_file.Get("HitTree")
        if not sig_tree or not bkg_tree:
            raise RuntimeError("HitTree not found")

        for v_id, v in variables.items():
            tag = f"{sensor_thickness}_{dataset}"
            h_sig_name = f"h_sig_{tag}_{v_id}"
            h_bkg_name = f"h_bkg_{tag}_{v_id}"

            h_sig = ROOT.TH1F(h_sig_name, f"{sensor_thickness} #mum", n_bins, v.xmin, v.xmax)
            h_bkg = ROOT.TH1F(h_bkg_name, f"{sensor_thickness} #mum", n_bins, v.xmin, v.xmax)

            sig_tree.Draw(f"{v_id}>>{h_sig_name}", "", "goff")
            bkg_tree.Draw(f"{v_id}>>{h_bkg_name}", "", "goff")

            normalize_in_place(h_sig)
            normalize_in_place(h_bkg)

            c = ROOT.TCanvas(f"c_{tag}_{v_id}", f"{v_id}", 800, 600)
            if v.yscale == "log":
                c.SetLogy(True)

            h_sig.SetLineColor(ROOT.kRed)
            h_bkg.SetLineColor(ROOT.kBlue)
            h_sig.SetLineWidth(2)
            h_bkg.SetLineWidth(2)

            h_sig.SetMaximum(v.ymax)
            h_sig.GetXaxis().SetTitle(v.label)
            h_sig.GetYaxis().SetTitle("Normalized entries")

            h_sig.Draw("HIST")
            h_bkg.Draw("HIST SAME")

            leg = ROOT.TLegend(0.7, 0.65, 0.9, 0.8) if v.legend == "right" else ROOT.TLegend(0.4, 0.65, 0.6, 0.8)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.AddEntry(h_sig, "Signal", "l")
            leg.AddEntry(h_bkg, "Background", "l")
            leg.Draw()

            c.SaveAs(f"{tag}_{v_id}_dist.png")
            out_file.cd()
            h_sig.Write()
            h_bkg.Write()
            c.Write()

        sig_file.Close()
        bkg_file.Close()
        out_file.Close()
