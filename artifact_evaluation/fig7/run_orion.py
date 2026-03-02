import os
import time
import argparse
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


num_runs = 1

# trace_files = [
    # ("", "", "profile", 160000),
#     # ("", "", "Rnet_32_Rnet_32", 160000),
#     # ("", "", "R1net_32_R1net_32", 160000),
#     # ("", "", "Dnet_32_Dnet_32", 160000),
#     # ("", "", "Vnet_32_Vnet_32", 160000),
#     #  ("", "", "Mnet_32_Mnet_32", 160000),
#     # ("", "", "Rnet_32_Rnet_32_Rnet_32_Rnet_32", 160000),
#     #  ("", "", "R1net_32_R1net_32_R1net_32_R1net_32", 160000),
#     # ("", "", "Dnet_32_Dnet_32_Dnet_32_Dnet_32", 160000),
#     # ("", "", "Vnet_32_Vnet_32_Vnet_32_Vnet_32", 160000),
#     #  ("", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),

#     #  ("", "", "Rnet_32_Rnet_16", 160000),
#     # ("", "", "Dnet_32_Dnet_16", 160000),
#     # ("", "", "Dnet_32_Rnet_16", 160000),
#     # ("", "", "Rnet_32_Dnet_32", 160000),
#     # ("", "", "Rnet_32_Rnet_32", 160000),

#     # ("", "", "Rnet_32_R1net_32", 160000),
#     # ("", "", "R1net_32_Rnet", 160000),
#     # ("", "", "R1net_32_Vnet_32", 160000),

#     # ("", "", "Dnet_32_Vnet_32", 160000),


#     #  ("", "", "Rnet_32_Rnet_16_Dnet_32_Dnet_16", 160000),
#     # ("", "", "Vnet_32_Rnet_32_Dnet_32_R1net32", 160000),
# ]




# trace_files = [
    # ("", "", "Rnet_8_Rnet_8", 160000),
    # ("", "", "R1net_8_R1net_8", 160000),
    #  ("", "", "Dnet_8_Dnet_8", 160000),
    #  ("", "", "Vnet_8_Vnet_8", 160000),
    # ("", "", "Mnet_32_Mnet_32", 160000),
    # ("", "", "Tnet_8_Tnet_8", 160000),

    # ("", "", "Rnet_8", 160000),
    # ("", "", "R1net_8", 160000),
    # ("", "", "Dnet_8", 160000),
    # ("", "", "Vnet_8", 160000),
    # ("", "", "Mnet_32", 160000),
    # ("", "", "Tnet_8", 160000),
    # ("", "", "Bnet_8", 160000),


    # ("", "", "Bnet_8_Bnet_8", 160000),
    # ("", "", "Rnet_8_Dnet_8", 160000),
    # ("", "", "Rnet_8_Vnet_8", 160000),
    # ("", "", "R1net_8_Dnet_8", 160000),
    # ("", "", "R1net_8_Vnet_8", 160000),

    # ("", "", "Rnet_8_R1net_8_Dnet_8_Vnet_8", 160000), 

    # ("", "", "Tnet_8_Tnet_8_Tnet_8_Tnet_8", 160000),
    # ("", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),
    # ("", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),
    # ("", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),
    # ("", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),
    #  ("", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),
# ]

trace_files = [
    # ("", "", "Rnet_8", 160000),
    #  ("", "", "Vnet_8", 160000),
    #   ("", "", "Tnet_8", 160000),

    # ("", "", "R1net_8", 160000),
    # ("", "", "Dnet_8", 160000),
    # ("", "", "Vnet_8", 160000),
    # ("", "", "Mnet_32", 160000),

    ("", "", "Rnet_8_Rnet_8", 160000),
    # ("", "", "R1net_8_R1net_8", 160000),
    # ("", "", "Dnet_8_Dnet_8", 160000),
    # ("", "", "Vnet_8_Vnet_8", 160000),
    # ("", "", "Mnet_32_Mnet_32", 160000),
    # ("", "", "Tnet_8_Tnet_8", 160000),

    
    #  ("", "", "Rnet_8", 160000),
    # ("", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),
    # ("", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),
    # ("", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),
    # ("", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),
    # ("", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),
    # ("", "", "Tnet_8_Tnet_8_Tnet_8_Tnet_8", 160000),

    # ("", "", "Mnet_32_R1net_8_Vnet_8_Dnet_8", 160000),
    # ("", "", "Rnet_8_Mnet_32_R1net_8_Dnet_8", 160000),
    # ("", "", "Rnet_8_Mnet_32_R1net_8_Vnet_8", 160000),
    # ("", "", "Rnet_8_Mnet_32_Vnet_8_Dnet_8", 160000),
    # ("", "", "Rnet_8_R1net_8_Vnet_8_Dnet_8", 160000),

    # ("", "", "Rnet_8_Mnet_32", 160000),
    # ("", "", "Rnet_8_R1net_8", 160000),
    # ("", "", "Rnet_8_Vnet_8", 160000),
    # ("", "", "Rnet_8_Dnet_8", 160000),
    # ("", "", "Mnet_32_R1net_8", 160000),
    # ("", "", "Mnet_32_Vnet_8", 160000),
    # ("", "", "Mnet_32_Dnet_8", 160000),
    # ("", "", "R1net_8_Vnet_8", 160000),
    # ("", "", "R1net_8_Dnet_8", 160000),
    # ("", "", "Vnet_8_Dnet_8", 160000),
    # ("", "", "Rnet_8_Bnet_8", 160000),
    # ("", "", "Mnet_32_Bnet_8", 160000),

    # ("", "", "Rnet_8_Tnet_8", 160000),
    # ("", "", "Mnet_32_Tnet_8", 160000),
    # ("", "", "R1net_8_Tnet_8", 160000),
    # ("", "", "Vnet_8_Tnet_8", 160000),
    # ("", "", "Dnet_8_Tnet_8", 160000),

    # ("", "", "R1net_8_Bnet_8", 160000), 
    # ("", "", "Vnet_8_Bnet_8", 160000),
    # ("", "", "Dnet_8_Bnet_8", 160000),

    # ("", "", "Tnet_8_Bnet_8", 160000),
]


for (be, hp, f, max_be_duration) in trace_files:
    for run in range(num_runs):
        print(be, hp, run, flush=True)
        # run
        # file_path = f"config_files/new_baselines/{f}.json"
        file_path = f"config_files/new_baselines/{f}.json"
        # file_path = f"config_files/fourMixedModel/{f}.json"
        print(file_path)
        # print(mymask)
        lib_path =  "/home/zixi/orion_bu/src/cuda_capture/libinttemp.so"
        # os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion_bu/src/cuda_capture/libinttemp.so' python3.8 ../../benchmarking/launch_jobs.py --algo orion --config_file {file_path} --orion_max_be_duration {max_be_duration}")
        os.system(f"LD_PRELOAD='{lib_path}' python3.8 ../../benchmarking/launch_jobs.py --algo orion --config_file {file_path} --orion_max_be_duration {max_be_duration}")
        # copy results
        # os.system(f"cp client_1.json results/orion_bu/{be}_{hp}_{run}_hp.json")
        # os.system(f"cp client_0.json results/orion_bu/{be}_{hp}_{run}_be.json")

        # os.system("rm client_1.json")
        # os.system("rm client_0.json")