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

loose  = -0.6
medium = -0.4
tight  = -0.2

eval_data_name = kfcv_tag + "_eval_score_data.npz"

eval_data = np.load(eval_data_name)
eval_sig_data = eval_data["eval_sig_scores"]
eval_bkg_data = eval_data["eval_bkg_scores"]

eval_sig_len = len(eval_sig_data)
eval_bkg_len = len(eval_bkg_data)

print(f"Total evaluation signal entries: {eval_sig_len}")
print(f"Total evaluation background entries: {eval_bkg_len}")

og_loose_eval_sig_mask = eval_sig_data >= og_loose
og_medium_eval_sig_mask = eval_sig_data >= og_medium
og_tight_eval_sig_mask = eval_sig_data >= og_tight

og_loose_eval_sig = eval_sig_data[og_loose_eval_sig_mask]
og_medium_eval_sig = eval_sig_data[og_medium_eval_sig_mask]
og_tight_eval_sig = eval_sig_data[og_tight_eval_sig_mask]

og_loose_eval_bkg_mask = eval_bkg_data >= og_loose
og_medium_eval_bkg_mask = eval_bkg_data >= og_medium
og_tight_eval_bkg_mask = eval_bkg_data >= og_tight

og_loose_eval_bkg = eval_bkg_data[og_loose_eval_bkg_mask]
og_medium_eval_bkg = eval_bkg_data[og_medium_eval_bkg_mask]
og_tight_eval_bkg = eval_bkg_data[og_tight_eval_bkg_mask]

print(f"\n\nOriginal loose cut (>= {og_loose}) results:")
print(f"Signal remaining ratio: {len(og_loose_eval_sig)/eval_sig_len}")
print(f"Background remaining ratio: {len(og_loose_eval_bkg)/eval_bkg_len}")

print(f"\nOriginal medium cut (>= {og_medium}) results:")
print(f"Signal remaining ratio: {len(og_medium_eval_sig)/eval_sig_len}")
print(f"Background remaining ratio: {len(og_medium_eval_bkg)/eval_bkg_len}")

print(f"\nOriginal tight cut (>= {og_tight}) results:")
print(f"Signal remaining ratio: {len(og_tight_eval_sig)/eval_sig_len}")
print(f"Background remaining ratio: {len(og_tight_eval_bkg)/eval_bkg_len}")

loose_eval_sig_mask = eval_sig_data >= loose
medium_eval_sig_mask = eval_sig_data >= medium
tight_eval_sig_mask = eval_sig_data >= tight

loose_eval_sig = eval_sig_data[loose_eval_sig_mask]
medium_eval_sig = eval_sig_data[medium_eval_sig_mask]
tight_eval_sig = eval_sig_data[tight_eval_sig_mask]

loose_eval_bkg_mask = eval_bkg_data >= loose
medium_eval_bkg_mask = eval_bkg_data >= medium
tight_eval_bkg_mask = eval_bkg_data >= tight

loose_eval_bkg = eval_bkg_data[loose_eval_bkg_mask]
medium_eval_bkg = eval_bkg_data[medium_eval_bkg_mask]
tight_eval_bkg = eval_bkg_data[tight_eval_bkg_mask]

print(f"\n\nNew loose cut (>= {loose}) results:")
print(f"Signal remaining ratio: {len(loose_eval_sig)/eval_sig_len}")
print(f"Background remaining ratio: {len(loose_eval_bkg)/eval_bkg_len}")

print(f"\nNew medium cut (>= {medium}) results:")
print(f"Signal remaining ratio: {len(medium_eval_sig)/eval_sig_len}")
print(f"Background remaining ratio: {len(medium_eval_bkg)/eval_bkg_len}")

print(f"\nNew tight cut (>= {tight}) results:")
print(f"Signal remaining ratio: {len(tight_eval_sig)/eval_sig_len}")
print(f"Background remaining ratio: {len(tight_eval_bkg)/eval_bkg_len/eval_bkg_len}")
