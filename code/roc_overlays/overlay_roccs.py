import ROOT

base_path = "/global/cfs/projectdirs/atlas/jashley/mjolnir/beta/code/temp_results/"

def get_kfcv_tag(sensor_thickness, min_node_size, max_depth, n_trees, beta):
    tag = (
        str(sensor_thickness) + "_" +
        str(min_node_size[0]) + "-" +
        str(min_node_size[2]) + "_" +
        str(max_depth) + "_" +
        str(n_trees) + "_" +
        str(beta[0]) + "-" +
        str(beta[2])
    )
    return tag

# Open files
f_tag_1 = "50_0-5_12_200_0-1"
f_tag_2 = "50_0-5_13_200_0-1"
f_tag_3 = "50_0-5_13_400_0-1"
f_tag_4 = "50_0-5_14_200_0-1"
f_tag_5 = "50_1-0_10_400_0-1"
f_tag_6 = "50_1-0_12_400_0-1"
f_tag_7 = "50_1-5_14_200_0-1"

f_name_1 = f_tag_1 + "_TMVACV.root"
f_name_2 = f_tag_2 + "_TMVACV.root"
f_name_3 = f_tag_3 + "_TMVACV.root"
f_name_4 = f_tag_4 + "_TMVACV.root"
f_name_5 = f_tag_5 + "_TMVACV.root"

root_f_1 = ROOT.TFile.Open(base_path + f_name_1)
root_f_2 = ROOT.TFile.Open(base_path + f_name_2)
root_f_3 = ROOT.TFile.Open(base_path + f_name_3)
root_f_4 = ROOT.TFile.Open(base_path + f_name_4)
root_f_5 = ROOT.TFile.Open(base_path + f_name_5)

# Get histograms
h_name = "datasetcv/Method_CrossValidation/BDT/MVA_BDT_rejBvsS"
h1 = root_f_1.Get(h_name)
h2 = root_f_2.Get(h_name)
h3 = root_f_3.Get(h_name)
h4 = root_f_4.Get(h_name)
h5 = root_f_5.Get(h_name)

# Keep histograms in memory
h1.SetDirectory(0)
h2.SetDirectory(0)
h3.SetDirectory(0)
h4.SetDirectory(0)
h5.SetDirectory(0)

root_f_1.Close()
root_f_2.Close()
root_f_3.Close()
root_f_4.Close()
root_f_5.Close()

c = ROOT.TCanvas("c", "Overlay", 800, 600)

# Restrict signal efficiency range
x_min, x_max = 0.85, 1

h1.GetXaxis().SetRangeUser(x_min, x_max)
h2.GetXaxis().SetRangeUser(x_min, x_max)
h3.GetXaxis().SetRangeUser(x_min, x_max)
h4.GetXaxis().SetRangeUser(x_min, x_max)
h5.GetXaxis().SetRangeUser(x_min, x_max)

# No stat box
h1.SetStats(0)
h2.SetStats(0)
h3.SetStats(0)
h4.SetStats(0)
h5.SetStats(0)

# Draw with colors and options
h1.SetLineColor(ROOT.kBlack)
h2.SetLineColor(ROOT.kBlue)
h3.SetLineColor(ROOT.kMagenta)
h4.SetLineColor(ROOT.kRed)
h5.SetLineColor(ROOT.kOrange)

h1.Draw("HIST")
h2.Draw("HIST SAME") 
h3.Draw("HIST SAME")
h4.Draw("HIST SAME")
h5.Draw("HIST SAME")

# Add legend
legend = ROOT.TLegend(0.12, 0.12, 0.35, 0.25)

legend.AddEntry(h1,f_tag_1,"l")
legend.AddEntry(h2,f_tag_2,"l")
legend.AddEntry(h3,f_tag_3,"l")
legend.AddEntry(h4,f_tag_4,"l")
legend.AddEntry(h5,f_tag_5,"l")

legend.Draw()

c.SaveAs("overlay_attempt_2.png")

# Save to ROOT file
out_file = ROOT.TFile.Open("overlay_attempt_2.root", "RECREATE")
out_file.cd()

h1.Write(f_tag_1)
h2.Write(f_tag_2)
h3.Write(f_tag_3)
h4.Write(f_tag_4)
h5.Write(f_tag_5)

c.Write("overlay")

out_file.Close()
