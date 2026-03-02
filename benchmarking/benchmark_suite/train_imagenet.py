#!/usr/bin/env python3
import os
import re
import csv
import json
import time
import random
import argparse
import threading
from ctypes import cdll
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torchvision import models, datasets, transforms
import multiprocessing as mp

torch.backends.cudnn.enabled = True
print(torch.backends.cudnn.enabled)
print(torchvision.__file__)


def seed_everything(seed: int):
    import random as _random
    import os as _os
    import numpy as _np

    _random.seed(seed)
    _os.environ["PYTHONHASHSEED"] = str(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DummyDataLoader:
    def __init__(self, batchsize: int):
        self.batchsize = batchsize
        self.data = torch.rand([self.batchsize, 3, 224, 224], pin_memory=True)
        self.target = torch.ones([self.batchsize], pin_memory=True, dtype=torch.long)

    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.target


class RealDataLoader:
    def __init__(self, batchsize: int):
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_dataset = datasets.ImageFolder(
            "/mnt/data/home/fot/imagenet/imagenet-raw-euwest4",
            transform=train_transform,
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batchsize, num_workers=8
        )

    def __iter__(self):
        print("Inside iter")
        return iter(self.train_loader)


def block(backend_lib, it):
    backend_lib.block(it)


def check_stop(backend_lib):
    return backend_lib.stop()


def pick_trace_sleep_times(
    trace_dir: str,
    num_iters: int,
    id_: Optional[int] = None,
    run_idx: Optional[int] = None,
    run_idx_is_1based: bool = True,
    loop_if_short: bool = True,
    shuffle: bool = False,
    trace_unit: str = "seconds",
) -> List[float]:
    """
    Return num_iters inter-arrival gaps (seconds) from one trace file in trace_dir.
    Selection priority:
      1) run_idx
      2) id_
      3) deterministic fallback: id_=0
    """
    trace_dir = Path(trace_dir)
    trace_files = sorted(p for p in trace_dir.iterdir() if p.is_file())
    if not trace_files:
        raise FileNotFoundError(f"No trace files found in {trace_dir}")

    n_files = len(trace_files)

    if run_idx is not None:
        r = int(run_idx)
        idx = (r - 1) if run_idx_is_1based else r
        idx = idx % n_files
        trace_file = trace_files[idx]
        select_reason = f"run_idx={r} -> trace_idx={idx}"
    elif id_ is not None:
        if not (0 <= id_ < n_files):
            raise IndexError(f"id_={id_} out of range (0..{n_files - 1})")
        trace_file = trace_files[id_]
        select_reason = f"id_={id_}"
    else:
        trace_file = trace_files[0]
        select_reason = "default trace_idx=0"

    with trace_file.open() as f:
        vals = [float(line.strip()) for line in f if line.strip()]

    if len(vals) < 2:
        raise ValueError(f"Trace file {trace_file} is too short")

    # If timestamps are monotone, convert to gaps
    if all(b >= a for a, b in zip(vals, vals[1:])):
        gaps = [vals[0]] + [b - a for a, b in zip(vals, vals[1:])]
    else:
        gaps = vals

    unit = trace_unit.strip().lower()
    if unit in ("milliseconds", "millisecond", "ms"):
        gaps = [g / 1.0 for g in gaps]
    elif unit in ("seconds", "second", "s"):
        pass
    else:
        raise ValueError(f"Unsupported trace_unit={trace_unit!r}")

    if len(gaps) < num_iters:
        if not loop_if_short:
            raise ValueError(f"Trace length {len(gaps)} < num_iters={num_iters}")
        reps = -(-num_iters // len(gaps))
        gaps = (gaps * reps)[:num_iters]

    if shuffle:
        random.shuffle(gaps)

    print(
        f"[pick_trace_sleep_times] picked trace file: {trace_file.name} "
        f"({select_reason}, unit={trace_unit} -> seconds)"
    )
    return gaps[:num_iters]


# =========================
# Run reservation helpers
# =========================
_RUN_RE = re.compile(r"run_(\d+)")


def _replace_last_run_token(s: str, new_run: int) -> str:
    ms = list(_RUN_RE.finditer(s))
    if not ms:
        return s
    m = ms[-1]
    return f"{s[:m.start()]}run_{new_run}{s[m.end():]}"


def _extract_run_from_csv_base(csv_base: Optional[str]) -> Optional[int]:
    if not csv_base:
        return None
    ms = list(_RUN_RE.finditer(str(csv_base)))
    if not ms:
        return None
    try:
        v = int(ms[-1].group(1))
        return v if v > 0 else None
    except Exception:
        return None


def _candidate_base_for_run(csv_base: str, run: int) -> str:
    base = str(csv_base)
    if _RUN_RE.search(base):
        return _replace_last_run_token(base, run)
    return f"{base}_run_{run}"


def reserve_group_run(csv_base: str, num_clients: int, start_run: int = 1) -> Tuple[str, int]:
    """
    Reserve ONE run id for the entire experiment:
    choose the first run where none of client_0..client_{num_clients-1} exists.
    """
    run = max(1, int(start_run))
    while True:
        candidate_base = _candidate_base_for_run(csv_base, run)
        occupied = any(
            os.path.exists(f"{candidate_base}_client_{tid}.csv")
            for tid in range(num_clients)
        )
        if not occupied:
            return candidate_base, run
        run += 1


def _resolve_run_idx_for_trace(run_idx: Optional[int], csv_base: Optional[str]) -> int:
    if run_idx is not None:
        try:
            r = int(run_idx)
            return r if r > 0 else 1
        except Exception:
            pass

    parsed = _extract_run_from_csv_base(csv_base)
    if parsed is not None:
        return parsed
    return 1


def imagenet_loop(
    model_name: str,
    batchsize: int,
    train: bool,
    num_iters: int,
    rps: float,
    uniform: bool,
    dummy_data: bool,
    local_rank: int,
    barriers,
    client_barrier,
    tid: int,
    input_file: str = "",
    rps_profile: str = "low",  # low | poisson | twitter | apollo
    algo_name: Optional[str] = None,
    trace_name: Optional[str] = None,
    run_idx: Optional[int] = None,
    csv_base: Optional[str] = None,
):
    seed_everything(42)
    print(model_name, batchsize, local_rank, tid)
    print(f"[imagenet_loop] rps_profile = {rps_profile}")
    if algo_name is not None:
        print(f"[imagenet_loop] algo = {algo_name}")
    if trace_name is not None:
        print(f"[imagenet_loop] trace = {trace_name}")
    if run_idx is not None:
        print(f"[imagenet_loop] run_idx = {run_idx}")
    if csv_base is not None:
        print(f"[imagenet_loop] csv_base = {csv_base}")

    backend_lib = cdll.LoadLibrary("/home/zixi/orion_bu/src/cuda_capture/libinttemp.so")

    # ---------- Request pattern ----------
    if rps_profile in ("twitter", "apollo"):
        if rps_profile == "twitter":
            trace_dir = "/home/zixi/orion_bu/benchmarking/benchmark_suite/TW_may25_1hr_traces/"
            trace_unit = "seconds"
        else:
            trace_dir = "/home/zixi/orion_bu/benchmarking/benchmark_suite/Real_world_trace/trace_txt"
            # trace_unit = "seconds"
            trace_unit = "millisecond"

        resolved_run_idx = _resolve_run_idx_for_trace(run_idx, csv_base)

        sleep_times = pick_trace_sleep_times(
            trace_dir=trace_dir,
            num_iters=num_iters,
            run_idx=resolved_run_idx,
            run_idx_is_1based=True,
            trace_unit=trace_unit,
        )
        print(
            f"[imagenet_loop] Using trace-based sleep times from {trace_dir} "
            f"(unit={trace_unit}, run_idx={resolved_run_idx})"
        )
    else:  
        
        # rps_map_low = {
        #     'resnet152':      15,
        #     'resnet101':      40,
        #     'densenet201':    15,
        #     'mobilenet_v2':   30,
        #     'vgg19':          28,
        # }
        rps_map_low = {
            'resnet152':      8,
            'resnet101':      8,
            'densenet201':    8,
            'mobilenet_v2':   8,
            'vgg19':          8,
        }

        # rps_map_low = {
        #     'resnet152':      25,
        #     'resnet101':      1,
        #     'densenet201':    30,
        #     'mobilenet_v2':   1,
        #     'vgg19':          1,
        # }
        if model_name not in rps_map_low:
            raise RuntimeError(f"Unknown model name: {model_name!r}; aborting.")

        base_rps = rps_map_low[model_name]

        if rps_profile == "poisson":
            sleep_times = np.random.exponential(
                scale=1.0 / base_rps, size=num_iters
            ).tolist()
            print(
                f"[imagenet_loop] Using Poisson arrivals with mean RPS={base_rps} for {model_name}"
            )
        elif rps_profile == "low":
            rps = base_rps
            sleep_times = [1.0 / rps] * num_iters
            print(f"[imagenet_loop] Using synthetic constant RPS={rps} for {model_name}")
        else:
            print(
                f"[imagenet_loop] WARNING: unknown rps_profile={rps_profile!r}, fallback to 'low'"
            )
            rps = base_rps
            sleep_times = [1.0 / rps] * num_iters
            print(f"[imagenet_loop] Using synthetic constant RPS={rps} for {model_name}")

    print(f"SIZE is {len(sleep_times)}")
    barriers[0].wait()

    print("-------------- thread id: ", threading.get_native_id())

    model = models.__dict__[model_name](num_classes=1000).to(0)

    if train:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss().to(local_rank)
    else:
        model.eval()

    train_loader = DummyDataLoader(batchsize) if dummy_data else RealDataLoader(batchsize)
    train_iter = enumerate(train_loader)

    batch_idx, batch = next(train_iter)
    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
    print("Enter loop!")

    next_startup = time.time()
    open_loop = True
    overall_start = time.time()
    start = time.time()

    timings = []

    for epoch in range(1):
        print("Start epoch:", epoch)
        while batch_idx < num_iters:
            start_iter = time.time()
            # time.sleep(1)
            if train:
                gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
                optimizer.zero_grad()
                output = model(gpu_data)
                loss = criterion(output, gpu_target)
                loss.backward()
                optimizer.step()

                block(backend_lib, batch_idx)
                timings.append(time.time() - start_iter)

                batch_idx, batch = next(train_iter)

                if batch_idx == 1:
                    barriers[0].wait()
                if batch_idx == 10:
                    barriers[0].wait()
                    start = time.time()

                if check_stop(backend_lib):
                    print("---- STOP!")
                    break

            else:
                with torch.no_grad():
                    cur_time = time.time()

                    if open_loop:
                        if cur_time >= next_startup:
                            print(f"Client {tid}, submit!, batch_idx={batch_idx}")
                            if batch_idx == 100:
                                torch.cuda.profiler.cudart().cudaProfilerStart()

                            _ = model(gpu_data)
                            block(backend_lib, batch_idx)

                            req_time = time.time() - next_startup
                            timings.append(req_time)
                            print(f"Client {tid} finished! It took {req_time}")

                            if batch_idx >= 10:
                                next_startup += sleep_times[batch_idx]
                            else:
                                next_startup = time.time()

                            batch_idx, batch = next(train_iter)

                            if batch_idx == 1 or batch_idx == 10:
                                barriers[0].wait()
                                if batch_idx == 10:
                                    next_startup = time.time()
                                    start = time.time()

                            dur = next_startup - time.time()
                            if dur > 0:
                                time.sleep(dur)

                            if check_stop(backend_lib):
                                print(f"Client {tid} ---- STOP!")
                                break
                    else:
                        print(f"Client {tid}, submit!, batch_idx={batch_idx}")
                        gpu_data = batch[0].to(local_rank)
                        _ = model(gpu_data)
                        block(backend_lib, batch_idx)
                        print(f"Client {tid} finished! Wait!")
                        batch_idx, batch = next(train_iter)
                        if batch_idx == 1 or batch_idx == 10:
                            barriers[0].wait()

    print(f"Client {tid} at barrier!")
    overall_end = time.time()
    overall_duration = overall_end - overall_start
    print(f"Client {tid} Total execution time: {overall_duration} seconds")
    barriers[0].wait()

    total_time = time.time() - start

    # Keep requests > 10
    timings = timings[10:]
    print(f"Client {tid} finished {len(timings)} iterations (requests > 10)")

    if not train and len(timings) > 0:
        p50 = np.percentile(timings, 50)
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)
        average = np.mean(timings)
        print(
            f"Client {tid} p50={p50} sec, p95={p95} sec, p99={p99} sec, avg={average}"
        )
        if total_time > 0:
            print(f"Client {tid} throughput ~ {(batch_idx - 10) / total_time} req/s")
    else:
        if total_time > 0:
            print(f"Client {tid} throughput ~ {(batch_idx - 10) / total_time} req/s")

    # =========================
    # SAVE TIMINGS TO CSV
    # fixed run base for all clients
    # =========================
    if csv_base is not None:
        csv_path = f"{csv_base}_client_{tid}.csv"
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        try:
            # x = create only, catch accidental overwrite
            with open(csv_path, "x", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["latency_s"])
                for v in timings:
                    writer.writerow([v])
            print(f"[Client {tid}] Wrote {len(timings)} latencies to {csv_path} (run_{run_idx})")
        except FileExistsError:
            print(f"[Client {tid}] ERROR: CSV already exists: {csv_path}")
        except Exception as e:
            print(f"[Client {tid}] ERROR writing CSV {csv_path}: {e}")
    else:
        csv_path = f"client_{tid}_timings.csv"
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["latency_s"])
                for v in timings:
                    writer.writerow([v])
            print(f"[Client {tid}] Wrote {len(timings)} latencies to {csv_path}")
        except Exception as e:
            print(f"[Client {tid}] ERROR writing CSV {csv_path}: {e}")

    print("Finished! Ready to join!")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_clients", type=int, default=4)
    p.add_argument("--model_name", type=str, default="resnet152")
    p.add_argument("--batchsize", type=int, default=8)
    p.add_argument("--num_iters", type=int, default=120)
    p.add_argument("--rps", type=float, default=0.0)
    p.add_argument("--uniform", action="store_true")
    p.add_argument("--dummy_data", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--rps_profile", type=str, default="low", choices=["low", "poisson", "twitter", "apollo"])
    p.add_argument("--algo_name", type=str, default=None)
    p.add_argument("--trace_name", type=str, default=None)
    p.add_argument("--run_idx", type=int, default=None)
    p.add_argument("--csv_base", type=str, default=None)
    p.add_argument("--input_file", type=str, default="")
    p.add_argument("--local_rank", type=int, default=0)
    return p.parse_args()


def build_client_configs(args):
    """
    If input_file is provided, expect list of client configs in JSON.
    Otherwise build homogeneous configs by args.
    """
    if args.input_file:
        with open(args.input_file, "r") as f:
            arr = json.load(f)

        cfgs = []
        for i, e in enumerate(arr):
            a = e.get("args", {})
            cfgs.append(
                dict(
                    model_name=a.get("model_name", args.model_name),
                    batchsize=int(a.get("batchsize", args.batchsize)),
                    train=bool(a.get("train", args.train)),
                    num_iters=int(e.get("num_iters", args.num_iters)),
                    rps=float(a.get("rps", args.rps)),
                    uniform=bool(a.get("uniform", args.uniform)),
                    dummy_data=bool(a.get("dummy_data", args.dummy_data)),
                )
            )
        return cfgs

    return [
        dict(
            model_name=args.model_name,
            batchsize=args.batchsize,
            train=args.train,
            num_iters=args.num_iters,
            rps=args.rps,
            uniform=args.uniform,
            dummy_data=args.dummy_data,
        )
        for _ in range(args.num_clients)
    ]


def main():
    args = parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    client_cfgs = build_client_configs(args)
    num_clients = len(client_cfgs)

    # Reserve ONE run id for all clients (fixes run split issue)
    chosen_run = args.run_idx
    group_csv_base = args.csv_base

    if args.csv_base:
        start_run = args.run_idx
        if start_run is None:
            parsed = _extract_run_from_csv_base(args.csv_base)
            start_run = parsed if parsed is not None else 1

        group_csv_base, chosen_run = reserve_group_run(
            csv_base=args.csv_base,
            num_clients=num_clients,
            start_run=start_run,
        )
        print(f"[main] Reserved run_{chosen_run}, group_csv_base={group_csv_base}")

    # shared barriers
    barrier = mp.Barrier(num_clients)
    client_barrier = mp.Barrier(num_clients)

    procs = []
    for tid, cfg in enumerate(client_cfgs):
        p = mp.Process(
            target=imagenet_loop,
            kwargs=dict(
                model_name=cfg["model_name"],
                batchsize=cfg["batchsize"],
                train=cfg["train"],
                num_iters=cfg["num_iters"],
                rps=cfg["rps"],
                uniform=cfg["uniform"],
                dummy_data=cfg["dummy_data"],
                local_rank=args.local_rank,
                barriers=[barrier],
                client_barrier=client_barrier,
                tid=tid,
                input_file=args.input_file,
                rps_profile=args.rps_profile,
                algo_name=args.algo_name,
                trace_name=args.trace_name,
                run_idx=chosen_run,
                csv_base=group_csv_base,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
