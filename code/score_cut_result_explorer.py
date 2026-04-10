import argparse
import numpy as np
import matplotlib.pyplot as plt

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
        "-s", required=True, type=str, help="MinNodeSize"    ) 
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

def get_kfcv_tag():
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

kfcv_tag = get_kfcv_tag()

og_loose  = -0.4
og_medium = -0.2
og_tight  = 0.0

eval_data_name = kfcv_tag + "_eval_score_data.npz"

eval_data = np.load(eval_data_name)
eval_sig_data = eval_data["eval_sig_scores"]
eval_bkg_data = eval_data["eval_bkg_scores"]

eval_sig_len = len(eval_sig_data)
eval_bkg_len = len(eval_bkg_data)

print(f"Total evaluation signal entries: {eval_sig_len}")
print(f"Total evaluation background entries: {eval_bkg_len}")

min_cut     = -0.2
max_cut     = -0.25
num_results = 25

cut_vals = np.linspace(min_cut, max_cut, num_results)

eval_sig_ratios = []
eval_bkg_ratios = []

for val in cut_vals:
    eval_sig_mask = eval_sig_data >= val
    eval_bkg_mask = eval_bkg_data >= val
     
    cut_eval_sig = eval_sig_data[eval_sig_mask]
    cut_eval_bkg = eval_bkg_data[eval_bkg_mask]

    sig_ratio = len(cut_eval_sig)/eval_sig_len
    bkg_ratio = len(cut_eval_bkg)/eval_bkg_len
     
    print(f"\n\n>= {val} cut results:")
    print(f"Signal remaining ratio: {len(cut_eval_sig)/eval_sig_len}")
    print(f"Background remaining ratio: {len(cut_eval_bkg)/eval_bkg_len}")

    eval_sig_ratios.append(sig_ratio)
    eval_bkg_ratios.append(bkg_ratio)

fig, ax = plt.subplots()

ax.plot(cut_vals, eval_sig_ratios, label="Signal")
ax.plot(cut_vals, eval_bkg_ratios, label="Background")

ax.set_xlabel("BDT score cut")
ax.set_ylabel("Ratio of remaining clusters")

ax.legend()
ax.grid()

fig.savefig("score_cut_ratios.pdf")
