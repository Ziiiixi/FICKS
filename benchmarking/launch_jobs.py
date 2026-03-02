import argparse
import json
import threading
import time
from ctypes import *
import os
import sys
from torchvision import models
import torch
import pynvml  # Import pynvml for energy measurement

home_directory = os.path.expanduser('~')
sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")
sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils")
from benchmark_suite.transformer_trainer import transformer_loop
sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/BERT")
from benchmark_suite.bert_trainer_mock import bert_loop

from benchmark_suite.train_imagenet import imagenet_loop
from benchmark_suite.toy_models.bnorm_trainer import bnorm_loop
from benchmark_suite.toy_models.conv_bn_trainer import conv_bn_loop

from src.scheduler_frontend import PyScheduler

function_dict = {
    "alexnet": imagenet_loop,               # mmy
    "resnet50": imagenet_loop,
    "resnet152": imagenet_loop,
    "resnet101": imagenet_loop,
    "mobilenet_v2": imagenet_loop,
    "densenet201": imagenet_loop,           # mmy
    "resnext50_32x4d": imagenet_loop,       # mmy
    "shufflenet_v2_x1_0": imagenet_loop,    # mmy
    "vgg19": imagenet_loop,                 # mmy
    "squeezenet1_0": imagenet_loop,         # mmy
    "bnorm": bnorm_loop,
    "conv_bnorm": conv_bn_loop,
    "bert": bert_loop,
    "transformer": transformer_loop,
    "vit_b_16": imagenet_loop,
}


def seed_everything(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def launch_jobs(config_dict_list, input_args, run_eval):
    print("in launch_jobs")
    seed_everything(42)

    print(config_dict_list)
    num_clients = len(config_dict_list)
    print(f"num_clients = {num_clients}")

    s = torch.cuda.Stream()

    # Initialize barriers
    num_barriers = num_clients + 1
    barriers = [threading.Barrier(num_barriers) for _ in range(num_clients)]
    client_barrier = threading.Barrier(num_clients)

    # Load scheduler library
    if run_eval:
        sched_lib = cdll.LoadLibrary("/home/zixi/orion_bu/src/scheduler/scheduler_eval.so")
    else:
        sched_lib = cdll.LoadLibrary("/home/zixi/orion_bu/src/scheduler/scheduler_eval.so")

    py_scheduler = PyScheduler(sched_lib, num_clients)

    print(torch.__version__)

    model_names = [config_dict['arch'] for config_dict in config_dict_list]
    model_files = [config_dict['kernel_file'] for config_dict in config_dict_list]

    additional_model_files = [
        config_dict['additional_kernel_file'] if 'additional_kernel_file' in config_dict else None
        for config_dict in config_dict_list
    ]
    num_kernels = [config_dict['num_kernels'] for config_dict in config_dict_list]
    num_iters = [config_dict['num_iters'] for config_dict in config_dict_list]
    train_list = [config_dict['args']['train'] for config_dict in config_dict_list]
    additional_num_kernels = [
        config_dict['additional_num_kernels'] if 'additional_num_kernels' in config_dict else None
        for config_dict in config_dict_list
    ]

    # Build CSV base path if trace_name & run_idx are provided
    csv_base = None
    if getattr(input_args, "trace_name", None) is not None and getattr(input_args, "run_idx", None) is not None:
        log_root = getattr(input_args, "log_root", "logs")
        csv_dir = os.path.join(log_root, input_args.algo, input_args.trace_name, input_args.rps_profile)
        os.makedirs(csv_dir, exist_ok=True)
        csv_base = os.path.join(csv_dir, f"run_{input_args.run_idx}")
        print(f"[launch_jobs] csv_base = {csv_base}")
    else:
        print("[launch_jobs] trace_name/run_idx not provided; CSVs will use default local paths.")

    tids = []
    threads = []
    for i, config_dict in enumerate(config_dict_list):
        func = function_dict[config_dict['arch']]
        model_args = config_dict['args']
        model_args.update({
            "num_iters": num_iters[i],
            "local_rank": 0,
            "barriers": barriers,
            "client_barrier": client_barrier,
            "tid": i
        })

        # Only for imagenet-based models: pass the rps_profile and CSV meta-info
        if func is imagenet_loop:
            model_args["rps_profile"] = input_args.rps_profile
            if csv_base is not None:
                model_args["csv_base"] = csv_base
                model_args["algo_name"] = input_args.algo
                model_args["trace_name"] = input_args.trace_name
                model_args["run_idx"] = input_args.run_idx

        thread = threading.Thread(target=func, kwargs=model_args)
        thread.start()
        tids.append(thread.native_id)
        threads.append(thread)

    print("TIDs:", tids)

    # ---- Decide which algorithm to run (maps to C++ int) ----
    algo_map = {
        'orion':       0,
        'reef':        1,
        'multistream': 2,
        'krisp':       3,
        'ficks':     4,
        'profile':     5,
    }
    algo_id = algo_map[input_args.algo]

    # One generic "algorithm parameter":
    if input_args.algo == 'reef':
        algo_param = input_args.reef_depth
    elif input_args.algo == 'orion':
        algo_param = input_args.orion_max_be_duration
    else:
        algo_param = 0

    print(f"[Scheduler] algo = {input_args.algo}, algo_id = {algo_id}, algo_param = {algo_param}")

    sched_thread = threading.Thread(
        target=py_scheduler.run_scheduler,
        args=(
            barriers,
            tids,
            model_names,
            model_files,
            additional_model_files,
            num_kernels,
            additional_num_kernels,
            num_iters,
            True,              # profile
            run_eval,
            algo_id,           # algorithm id
            algo_param,        # algorithm-specific parameter
            input_args.orion_hp_limit,
            input_args.orion_start_update,
            train_list,
            # thresholds:
            input_args.td1,
            input_args.td2,
        )
    )

    print("before start !!!!!!!!!!!!!!")
    sched_thread.start()

    # Start energy monitoring (currently only placeholder list)
    stop_event = threading.Event()
    power_list = []

    for thread in threads:
        thread.join()

    print("Train threads joined!")

    sched_thread.join()
    print("Scheduler thread joined!")

    # Stop energy monitoring
    stop_event.set()

    print("--------- All threads joined!")

    # Calculate total energy consumed
    total_energy_joules = sum(power_list) * 1.0  # power_list sampled every 1 second
    total_energy_kj = total_energy_joules / 1000.0  # Convert to kJ
    print(f"Total energy consumed: {total_energy_kj:.2f} kJ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algo',
        type=str,
        choices=['orion', 'reef', 'multistream', 'krisp', 'ficks', 'profile'],
        required=True,
        help='Choose one of orion | reef | multistream | krisp | ficks | profile'
    )
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the experiment configuration file')
    parser.add_argument('--reef_depth', type=int, default=1,
                        help='If reef is used, this stands for the queue depth')
    parser.add_argument('--orion_max_be_duration', type=int, default=1,
                        help='If orion is used, the maximum aggregate duration of on-the-fly best-effort kernels')
    parser.add_argument('--orion_start_update', type=int, default=1,
                        help='If orion is used, and the high priority job is training, this is the kernel id after which the update phase starts')
    parser.add_argument('--orion_hp_limit', type=int, default=1,
                        help='If orion is used, and the high priority job is training, this shows the maximum tolerated training iteration time')

    # New: thresholds for profile_check logic
    parser.add_argument('--td1', type=float, default=1.2,
                        help='TIME_COR_AVAIL < td1 * TIME_IND_OPT threshold')
    parser.add_argument('--td2', type=float, default=1.1,
                        help='TIME_IND_OPT vs TIME_COR_OPT * td2 threshold')

    parser.add_argument(
        '--rps_profile',
        type=str,
        choices=['low', 'high', 'twitter', 'apollo', 'poisson'],
        default='low',
        help='Request pattern: low | high | twitter | apollo | poisson'
    )

    # NEW: info for CSV naming (algo / trace / rps / run)
    parser.add_argument('--trace_name', type=str, default=None,
                        help='Trace / config name (e.g., Rnet_8_Rnet_8)')
    parser.add_argument('--run_idx', type=int, default=None,
                        help='Run index for repeated experiments')
    parser.add_argument('--log_root', type=str, default='logs',
                        help='Root directory for logs/CSVs (default: logs)')

    args = parser.parse_args()

    torch.cuda.set_device(0)
    profile = True
    with open(args.config_file) as f:
        config_dict = json.load(f)
    launch_jobs(config_dict, args, True)
