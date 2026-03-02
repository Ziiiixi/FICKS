import torch
import threading
import time
import modeling
import numpy as np
import json
import threading

cuda_lock = threading.Lock()
from optimization import BertAdam

from ctypes import *
import os

def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.input_ids = torch.ones((self.batchsize, 384), pin_memory=True).to(torch.int64)
        self.segment_ids = torch.ones((self.batchsize, 384), pin_memory=True).to(torch.int64)
        self.input_mask = torch.ones((self.batchsize, 384), pin_memory=True).to(torch.int64)
        self.start_positions = torch.zeros((self.batchsize,), pin_memory=True).to(torch.int64)
        self.end_positions = torch.ones((self.batchsize,), pin_memory=True).to(torch.int64)


    def __iter__(self):
        return self

    def __next__(self):
        return self.input_ids, self.segment_ids, self.input_mask, self.start_positions, self.end_positions


def block(backend_lib, it):
    # block client until request served
    backend_lib.block(it)


def check_stop(backend_lib):
    return backend_lib.stop()

from pathlib import Path
import random
from typing import List, Optional

def pick_trace_sleep_times(trace_dir: str,
                           num_iters: int,
                           id_: Optional[int] = None,
                           loop_if_short: bool = True,
                           shuffle: bool = False) -> List[float]:
    """
    Return *num_iters* inter-arrival gaps (seconds) taken from the id_-th trace
    file in *trace_dir* (alphabetical order).

    Parameters
    ----------
    trace_dir : str
        Directory containing trace text files (one float per line).
    num_iters : int
        Number of sleep times to return.
    id_ : int | None
        0-based index into sorted(trace_files).
        • id_ = 0 → first file, id_ = 1 → second file, …
        • If None, choose a random file (back-compatibility).
    loop_if_short : bool, default True
        Repeat the trace if it is shorter than *num_iters*.
    shuffle : bool, default False
        Shuffle the resulting list after looping.
    """
    trace_dir = Path(trace_dir)
    trace_files = sorted(p for p in trace_dir.iterdir() if p.is_file())
    if not trace_files:
        raise FileNotFoundError(f"No trace files found in {trace_dir}")

    # ----- file selection ----------------------------------------------------
    if id_ is None:
        trace_file = random.choice(trace_files)
    else:
        if not (0 <= id_ < len(trace_files)):
            raise IndexError(f"id_={id_} out of range (0‒{len(trace_files)-1})")
        trace_file = trace_files[id_]

    # ----- read and convert to gaps -----------------------------------------
    with trace_file.open() as f:
        vals = [float(line.strip()) for line in f if line.strip()]

    if len(vals) < 2:
        raise ValueError(f"Trace file {trace_file} is too short")

    # If monotone non-decreasing ⇒ absolute timestamps → convert to gaps
    if all(b >= a for a, b in zip(vals, vals[1:])):
        gaps = [vals[0]] + [b - a for a, b in zip(vals, vals[1:])]
    else:
        gaps = vals

    # ----- extend / shuffle --------------------------------------------------
    if len(gaps) < num_iters:
        if not loop_if_short:
            raise ValueError(f"Trace length {len(gaps)} < num_iters={num_iters}")
        reps = -(-num_iters // len(gaps))          # ceil division
        gaps = (gaps * reps)[:num_iters]

    if shuffle:
        random.shuffle(gaps)

    return gaps[:num_iters]

def bert_loop(model_name, batchsize, train, num_iters, rps, uniform, dummy_data, local_rank, barriers, client_barrier, tid):

    seed_everything(42)
    backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/orion_bu/src/cuda_capture/libinttemp.so")

    if rps > 0:
        # if uniform:
        #     sleep_times = [1/rps]*num_iters
        # else:
        # sleep_times = pick_trace_sleep_times(
        # # trace_dir="/home/zixi/orion_bu/benchmarking/benchmark_suite/TW_may25_1hr_traces/",
        # trace_dir="/home/zixi/orion_bu/benchmarking/benchmark_suite/Real_world_trace/trace_txt",
        # num_iters=num_iters,
        # id_=tid
        # )
        sleep_times = np.random.exponential(scale=1/20, size=num_iters)
    else:
        sleep_times = [0]*num_iters

    barriers[0].wait()
    
    if (train and tid==1):
        time.sleep(5)
        

    if (not train):
        model_config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "output_all_encoded_layers": False,
            "type_vocab_size": 2,
            "vocab_size": 30522
        }
    else:
        model_config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30522
        }

    config = modeling.BertConfig.from_dict(model_config)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    print("-------------- thread id:  ", threading.get_native_id())

    model = modeling.BertForQuestionAnswering(config).to(0)

    if train:
        model.train()
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=100)
    else:
        model.eval()

    train_loader = DummyDataLoader(batchsize)
    train_iter = enumerate(train_loader)
    batch_idx, batch = next(train_iter)
    
    #  open loop
    timings = []
    next_startup = time.time()
    open_loop = True


    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
                                
        while batch_idx < num_iters:
            with torch.no_grad():
                cur_time = time.time()
                #### OPEN LOOP ####
                if open_loop:
                    if (cur_time >= next_startup):
                        print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
                        if batch_idx==50:
                            torch.cuda.profiler.cudart().cudaProfilerStart()
                        input_ids, segment_ids, input_mask = batch[0].to(local_rank), batch[1].to(local_rank), batch[2].to(local_rank)
                        with cuda_lock:
                            output = model(input_ids, segment_ids, input_mask)
                        block(backend_lib, batch_idx)
                        req_time = time.time()-next_startup
                        timings.append(req_time)
                        print(f"Client {tid} finished! Wait! It took {req_time}")
                        if batch_idx>=10:
                            next_startup += sleep_times[batch_idx]
                        else:
                            next_startup = time.time()
                        batch_idx,batch = next(train_iter)
                        if ((batch_idx == 1) or (batch_idx == 10)):
                            barriers[0].wait()
                            if (batch_idx==10):
                                #time.sleep(1)
                                next_startup = time.time()
                                start = time.time()
                        dur = next_startup-time.time()
                        if (dur>0):
                            time.sleep(dur)
                        if check_stop(backend_lib):
                            print("---- STOP!")
                            break

                else:
                    ### CLOSED LOOP ###
                    print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
                    input_ids, segment_ids, input_mask = batch[0].to(local_rank), batch[1].to(local_rank), batch[2].to(local_rank)
                    output = model(input_ids, segment_ids, input_mask)
                    print(f"Client {tid} finished! Wait!")
                    if ((batch_idx == 1) or (batch_idx == 10)):
                        barriers[0].wait()
                    batch_idx,batch = next(train_iter)


    torch.cuda.profiler.cudart().cudaProfilerStop()
    barriers[0].wait()
    total_time = time.time() - start


    if not train and len(timings)>0:
            
        p50 = np.percentile(timings, 50)
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)
        average = np.mean(timings)
        print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec, average: {average}")
        data = {
            'p50_latency': p50*1000,
            'p95_latency': p95*1000,
            'p99_latency': p99*1000,
            'throughput': (batch_idx-10)/total_time
        }
    else:
        data = {
            'throughput': (batch_idx-10)/total_time
        }
    with open(f'client_{tid}.json', 'w') as f:
        json.dump(data, f)

    print("Finished! Ready to join!")
