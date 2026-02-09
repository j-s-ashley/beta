import ROOT
from pathlib import Path

ROOT.gROOT.SetBatch(True)   # no GUI
ROOT.TH1.AddDirectory(False)  # reduce ROOT auto-ownership surprises

# Optional: choose your own style
ROOT.gStyle.SetOptStat(0)

# Example: provide some line colors (ROOT color ints)
COLORS = [ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2, ROOT.kMagenta+2, ROOT.kOrange+7]

def hist_path(var_id: str, cls: str) -> str:
    # cls is "Signal" or "Background"
    return f"dataset/InputVariables_Id/{var_id}__{cls}_Id"

def fetch_hist(tfile: ROOT.TFile, var_id: str, cls: str, tag: str) -> ROOT.TH1:
    obj = tfile.Get(hist_path(var_id, cls))
    if not obj:
        raise KeyError(f"Missing hist: {tfile.GetName()} :: {hist_path(var_id, cls)}")
    if not isinstance(obj, ROOT.TH1):
        raise TypeError(f"Object is not a TH1: {tfile.GetName()} :: {hist_path(var_id, cls)}")

    # Clone to detach from file, and give it a unique name to avoid collisions
    h = obj.Clone(f"h_{var_id}_{cls}_{tag}")
    h.SetDirectory(0)
    return h

def normalize(h: ROOT.TH1) -> None:
    integ = h.Integral(0, h.GetNbinsX() + 1)  # include under/overflow if you want
    if integ > 0:
        h.Scale(1.0 / integ)

def style_hist(h: ROOT.TH1, color: int) -> None:
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.SetFillStyle(0)  # no fill (clean overlays)

def draw_overlay(var_id: str, var_cfg, files_map: dict, cls: str, outdir: str,
                 do_normalize: bool = True) -> None:
    # Open files once for this plot (cheap), or pre-open globally if you prefer
    tfiles = {}
    try:
        for i, (thick, fpath) in enumerate(files_map.items()):
            tf = ROOT.TFile.Open(fpath, "READ")
            if not tf or tf.IsZombie():
                raise OSError(f"Could not open {fpath}")
            tfiles[thick] = tf

        # Fetch + prep hists
        hists = []
        for i, (thick, tf) in enumerate(tfiles.items()):
            h = fetch_hist(tf, var_id, cls, tag=thick)
            if do_normalize:
                normalize(h)
            style_hist(h, COLORS[i % len(COLORS)])
            hists.append((thick, h))

        # Canvas
        c = ROOT.TCanvas(f"c_{var_id}_{cls}", "", 900, 700)
        c.SetTicks(1, 1)

        # Log y if requested
        if getattr(var_cfg, "yscale", "linear") == "log":
            c.SetLogy(True)

        # Apply x-range / y-max (ymax is in your config; interpret as post-normalization)
        # Note: set range before drawing to affect axes
        if getattr(var_cfg, "xmin", None) is not None and getattr(var_cfg, "xmax", None) is not None:
            for _, h in hists:
                h.GetXaxis().SetRangeUser(var_cfg.xmin, var_cfg.xmax)

        # Draw first hist to define axes
        thick0, h0 = hists[0]
        h0.SetTitle("")
        h0.GetXaxis().SetTitle(var_cfg.label)
        h0.GetYaxis().SetTitle("Normalized entries" if do_normalize else "Entries")

        # y-axis limits
        if getattr(var_cfg, "ymax", None) is not None:
            # If log scale, ensure a sensible minimum too
            if getattr(var_cfg, "yscale", "linear") == "log":
                h0.SetMinimum(1e-6)  # adjust if needed
            h0.SetMaximum(var_cfg.ymax)

        h0.Draw("HIST")

        # Draw the rest
        for thick, h in hists[1:]:
            h.Draw("HIST SAME")

        # Legend
        leg = ROOT.TLegend(0.62, 0.70, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        for thick, h in hists:
            leg.AddEntry(h, thick, "l")
        leg.Draw()

        # Save
        Path(outdir).mkdir(parents=True, exist_ok=True)
        out = Path(outdir) / f"{var_id}__{cls}.png"
        c.SaveAs(str(out))

    finally:
        # Close files
        for tf in tfiles.values():
            tf.Close()

def make_all_overlays(variables: dict, thickness_files: dict, outdir: str = "overlays"):
    for var_id, var_cfg in variables.items():
        # Signal plot
        draw_overlay(var_id, var_cfg, thickness_files, cls="Signal", outdir=outdir)
        # Background plot
        draw_overlay(var_id, var_cfg, thickness_files, cls="Background", outdir=outdir)

# Example usage:
# make_all_overlays(variables, thickness_files, outdir="tmva_overlays")

