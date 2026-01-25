import re
import os
import math
import ROOT
import argparse
import pyLCIO
import numpy as np
from pathlib import Path
from pyLCIO import EVENT, UTIL
from array import array

COLLECTIONS = [
        #"ITBarrelHits",
        #"ITEndcapHits",
        #"OTBarrelHits",
        #"OTEndcapHits",
        "VXDBarrelHits",
        #"VXDEndcapHits"
        ]

def options():
    parser = argparse.ArgumentParser(description="Generate BIB output hits TTree root file from input slcio file.")
    parser.add_argument("-i", required=True, type=Path, help="Input LCIO file")
    parser.add_argument(
        "--nhits",
        default=9,
        type=int,
        help="Max number of hits to dump for each collection",
    )
    return parser.parse_args()

def get_RMS(cluster_pos, p_hits, axis = ""):
    rms     = 0.
    sum_num = 0.
    sum_den = 0.
    local   = 0.
    
    for i in range(0, len(p_hits)):
        hit_constituent = p_hits[i]
        local_pos = hit_constituent.getPosition()
        if axis == "x":
            local = local_pos[0]
        else:
            local = local_pos[1]
        sum_num = sum_num + p_hits[i].getEDep() * ((cluster_pos - local)**2)
        sum_den = sum_den + p_hits[i].getEDep()

    rms = (sum_num / sum_den)
    return rms

def get_cov(cluster_posx, cluster_posy, p_hits):
    cov      = 0.
    sum_num  = 0.
    sum_den  = 0.
    local_x  = 0.
    local_y  = 0.

    for i in range(0, len(p_hits)):
        hit_constituent = p_hits[i]
        local_pos = hit_constituent.getPosition()
        local_x   = local_pos[0]
        local_y   = local_pos[1]
        sum_num   = sum_num + p_hits[i].getEDep() * ((cluster_posx - local_x) * (cluster_posy - local_y))
        sum_den   = sum_den + p_hits[i].getEDep()

    cov = (sum_num / sum_den)
    return cov

def get_skew(cluster_pos, p_hits, axis = ""):
    rms     = get_RMS(cluster_pos, p_hits, axis)
    sum_num = 0.
    sum_den = 0.
    local   = 0.
    skew    = 0.
    for i in range(0, len(p_hits)):
        hit_constituent = p_hits[i]
        local_pos = hit_constituent.getPosition()
        if axis == "x":
            local = local_pos[0]
        else:
            local = local_pos[1]

        sum_num = sum_num + p_hits[i].getEDep()*((cluster_pos-local)**3)
        sum_den = sum_den + p_hits[i].getEDep()*(math.sqrt(rms)**3)

    skew = (sum_num/sum_den)
    return skew

def get_theta(r, z):
    angle = math.atan2(r, z)
    return angle

def get_cluster_size(hits):
    y_max = -1e6
    x_max = -1e6
    y_min = 1e6
    x_min = 1e6

    for j in range(len(hits)):
        hit_constituent = hits[j]
        local_pos = hit_constituent.getPosition()
        x_local   = local_pos[0]
        y_local   = local_pos[1]

        if y_local < y_min:
            y_min = y_local
        if y_local > y_max:
            y_max = y_local

        if x_local < x_min:
            x_min = x_local
        if x_local > x_max:
            x_max = x_local

    cluster_size_y = (y_max - y_min) + 1
    cluster_size_x = (x_max - x_min) + 1

    return cluster_size_x, cluster_size_y

def get_collection(event, name):
    names = event.getCollectionNames()
    if name in names:
        return event.getCollection(name)
    return []

c = 299.792458 # mm/ns, used for corrected timing

def main():
    ops = options()
    in_file = ops.i
    print(f"Reading file {in_file}")

    stem      = in_file.stem #gets the name without extension
    in_ind    = stem.removeprefix("digiGNN_") #in_file.stem.split('_')[-1]
    out_dir   = "/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta"
    out_file  = f"{out_dir}/Hits_TTree_{in_ind}.root"
    tree_name = "HitTree"
    # Create a new ROOT file and TTree
    root_file = ROOT.TFile(out_file, "RECREATE")
    tree      = ROOT.TTree(tree_name, "Tree storing hit information")
    
    # Define cluster-level variables to be stored in the tree
    cluster_energy = array('f', [0.])
    cluster_time   = array('f', [0.])
    cluster_size   = array('f', [0.])
    cluster_x      = array('f', [0.])
    cluster_y      = array('f', [0.])
    cluster_z      = array('f', [0.])
    cluster_r      = array('f', [0.])
    cluster_theta  = array('f', [0.])
    cluster_x_size = array('f', [0.])
    cluster_y_size = array('f', [0.])
    cluster_rms_x  = array('f', [0.])
    cluster_rms_y  = array('f', [0.])
    cluster_skew_x = array('f', [0.])
    cluster_skew_y = array('f', [0.])
    cluster_aspect = array('f', [0.])
    cluster_ecc    = array('f', [0.])

    max_n_pix = ops.nhits
    pixel_E = [array('f', [0.]) for _ in range(max_n_pix)]
    pixel_T = [array('f', [0.]) for _ in range(max_n_pix)]
    for i in range(max_n_pix):
        tree.Branch(f"PixelHits_EnergyDeposited_{i}", pixel_E[i], f"PixelHits_EnergyDeposited_{i}/F")
        tree.Branch(f"PixelHits_ArrivalTime_{i}", pixel_T[i], f"PixelHits_ArrivalTime_{i}/F")

    # Create branches
    tree.Branch("Cluster_EnergyDeposited", cluster_energy, "Cluster_EnergyDeposited/F")
    tree.Branch("Cluster_ArrivalTime", cluster_time, "Cluster_ArrivalTime/F")
    tree.Branch("Cluster_Size_tot", cluster_size, "Cluster_Size_tot/F")
    tree.Branch("Cluster_Size_x", cluster_x_size, "Cluster_Size_x/F")
    tree.Branch("Cluster_Size_y", cluster_y_size, "Cluster_Size_y/F")
    tree.Branch("Cluster_x", cluster_x, "Cluster_x/F")
    tree.Branch("Cluster_y", cluster_y, "Cluster_y/F")
    tree.Branch("Cluster_z", cluster_z, "Cluster_z/F")
    tree.Branch("Cluster_r", cluster_r, "Cluster_r/F")
    tree.Branch("Incident_Angle", cluster_theta, "Incident_Angle/F")
    tree.Branch("Cluster_RMS_x", cluster_rms_x, "Cluster_RMS_x/F")
    tree.Branch("Cluster_RMS_y", cluster_rms_y, "Cluster_RMS_y/F")
    tree.Branch("Cluster_Skew_x", cluster_skew_x, "Cluster_Skew_x/F")
    tree.Branch("Cluster_Skew_y", cluster_skew_y, "Cluster_Skew_y/F")
    tree.Branch("Cluster_AspectRatio", cluster_aspect, "Cluster_AspectRatio/F")
    tree.Branch("Cluster_Eccentricity", cluster_ecc, "Cluster_Eccentricity/F")
    # TODO : Subdetectors
    
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(str(in_file))

    count_tot = 0
    count_l0  = 0

    # Start of event loop
    for i_event, event in enumerate(reader):
        cols = {}
        for col in COLLECTIONS:
            cols[col] = get_collection(event, col)

        print(f"Event {i_event} has")
        for col in cols:
            event_count = len(cols[col])
            count_tot   = count_tot + event_count
            print(f"  {event_count:5} hits in {col}")

        # Within each event, get hit collections
        for col_name in COLLECTIONS:
            collection = cols[col_name]
            enc = collection.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding) # Get CellID info
            dec = UTIL.BitField64(enc) # Create 64-bit CellID decoder

            # Start loop over each hit in collection
            for i_hit, hit in enumerate(collection):
                dec.setValue(hit.getCellID0() | (hit.getCellID1() << 32)) # 32-bit to 64-bit, then shift
                layer_val = dec["layer"].value()
                if layer_val != 0:
                    continue

                count_l0 = count_l0 + 1

                position = hit.getPosition()
                x_pos    = position[0]
                y_pos    = position[1]
                z_pos    = position[2]
                r_pos    = math.sqrt(x_pos**2+y_pos**2)

                cluster_energy[0] = hit.getEDep()
                
                # Compute corrected timing
                t_raw             = hit.getTime()
                d_pos             = math.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
                t_cor             = t_raw - (d_pos / c)
                cluster_time[0]   = t_cor
                
                cluster_x[0]      = x_pos
                cluster_y[0]      = y_pos
                cluster_z[0]      = z_pos
                cluster_r[0]      = r_pos
                cluster_theta[0]  = get_theta(r_pos, z_pos)

                pixel_hits = hit.getRawHits()
                n_pix      = len(pixel_hits)
                cluster_xhits, cluster_yhits = get_cluster_size(pixel_hits)
                cluster_size[0]   = n_pix
                cluster_x_size[0] = cluster_xhits
                cluster_y_size[0] = cluster_yhits
                cluster_rms_x[0]  = get_RMS(x_pos, pixel_hits,"x")
                cluster_rms_y[0]  = get_RMS(y_pos, pixel_hits,"y")
                cluster_skew_x[0] = get_skew(x_pos, pixel_hits,"x")
                cluster_skew_y[0] = get_skew(y_pos, pixel_hits,"y")

                # Compute eigenvalues of covariance matrix
                cluster_xy = get_cov(x_pos, y_pos, pixel_hits)
                trace      = cluster_rms_x[0] + cluster_rms_y[0]
                det        = cluster_rms_x[0] * cluster_rms_y[0] - cluster_xy**2
                lambda_1   = (trace + math.sqrt(trace**2 - 4*det)) / 2
                lambda_2   = (trace - math.sqrt(trace**2 - 4*det)) / 2

                if lambda_1 >= lambda_2:
                    if lambda_2 > 0:
                        cluster_aspect[0] = math.sqrt(lambda_1 / lambda_2)
                    else:
                        cluster_aspect[0] = 0.
                    cluster_ecc[0] = math.sqrt(1 - (lambda_2 / lambda_1))
                else:
                    if lambda_1 > 0:
                        cluster_aspect[0] = math.sqrt(lambda_2 / lambda_1)
                    else:
                        cluster_aspect[0] = 0.
                    cluster_ecc[0] = math.sqrt(1 - (lambda_1 / lambda_2))

                pix_list = [pixel_hits[i].getEDep() for i in range(pixel_hits.size())]
                nh       = min(max_n_pix, len(pix_list))
                pix_ind  = np.argsort(pix_list)[-nh:][::-1]
                for j in range(max_n_pix):
                    pixel_E[j][0] = 0.
                    pixel_T[j][0] = 0.
                for j, ind in enumerate(pix_ind):
                    pixel_E[j][0] = pixel_hits[int(ind)].getEDep()
                    pixel_T[j][0] = pixel_hits[int(ind)].getTime()

                tree.Fill()
                    
    # Write and close
    tree.Write()
    root_file.Close()

    print(f"Total VXB clusters {count_tot} and on layer0 {count_l0}")
    print(f"ROOT file '{out_file}' with tree '{tree_name}' created.")

if __name__ == "__main__":
    main()
