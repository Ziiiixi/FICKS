import ctypes
from ctypes import *
import torch
import numpy as np
import os
import time


class PyScheduler:

    def __init__(self, sched_lib, num_clients):
        torch.cuda.set_device(0)
        self._scheduler = sched_lib.sched_init()
        self._sched_lib = sched_lib
        self._num_clients = num_clients

    def run_scheduler(
        self,
        barriers,
        tids,
        model_names,
        kernel_files,
        additional_kernel_files,
        num_kernels,
        additional_num_kernels,
        num_iters,
        profile,
        run_eval,
        algo_id,       # 0=orion, 1=reef, 2=multistream, 3=krisp, 4=ficks/reserve
        algo_param,    # reef_depth / orion_max_be_duration / etc.
        hp_limit,
        update_start,
        train,
        td1=1.2,
        td2=1.1,
    ):
        # print(f"[PyScheduler] algo_id={algo_id}, algo_param={algo_param}, td1={td1}, td2={td2}")

        model_names_ctypes = [x.encode('utf-8') for x in model_names]
        lib_names = [x.encode('utf-8') for x in kernel_files]

        # ----- convert arrays -----
        IntAr = c_int * self._num_clients
        tids_ar = IntAr(*tids)
        num_kernels_ar = IntAr(*num_kernels)
        num_iters_ar = IntAr(*num_iters)

        CharAr = c_char_p * self._num_clients
        model_names_ctypes_ar = CharAr(*model_names_ctypes)
        lib_names_ar = CharAr(*lib_names)

        BoolAr = c_bool * self._num_clients
        train_ar = BoolAr(*train)

        # print("[PyScheduler] train flags:", train)

        # ----- ctypes signatures -----

        # void setup(Scheduler*,
        #            int num_clients,
        #            int* tids,
        #            char** models,
        #            char** files,
        #            int* num_kernels,
        #            int* num_iters,
        #            bool* train,
        #            int algo_id,
        #            float td1,
        #            float td2);
        self._sched_lib.setup.argtypes = [
            c_void_p,              # Scheduler*
            c_int,                 # num_clients
            POINTER(c_int),        # tids
            POINTER(c_char_p),     # models
            POINTER(c_char_p),     # files
            POINTER(c_int),        # num_kernels
            POINTER(c_int),        # num_iters
            POINTER(c_bool),       # train
            c_int,                 # algo_id
            c_float,               # td1
            c_float,               # td2
        ]
        self._sched_lib.setup.restype = None

        # void* schedule(Scheduler*,
        #                int num_clients,
        #                bool profile_mode,
        #                int iter,
        #                bool warmup,
        #                int warmup_iters,
        #                int algo_id,
        #                int algo_param,
        #                int hp_limit,
        #                int update_start);
        self._sched_lib.schedule.argtypes = [
            c_void_p,   # Scheduler*
            c_int,      # num_clients
            c_bool,     # profile_mode
            c_int,      # iter
            c_bool,     # warmup
            c_int,      # warmup_iters
            c_int,      # algo_id
            c_int,      # algo_param (reef_depth / orion_max_be_duration / etc.)
            c_int,      # hp_limit
            c_int,      # update_start
        ]
        self._sched_lib.schedule.restype = c_void_p

        # void setup_change(Scheduler*, int client_id, const char* new_file,
        #                   int new_num_kernels, int algo_id);
        self._sched_lib.setup_change.argtypes = [
            c_void_p,      # Scheduler*
            c_int,         # client_id
            c_char_p,      # new kernel file
            c_int,         # new_num_kernels
            c_int,         # algo_id
        ]
        self._sched_lib.setup_change.restype = None

        # print("[PyScheduler] models:", model_names)
        # print("[PyScheduler] libs:", lib_names)
        # print("[PyScheduler] tids:", tids)

        # ---- call setup (inject thresholds + algo) ----
        self._sched_lib.setup(
            self._scheduler,
            self._num_clients,
            tids_ar,
            model_names_ctypes_ar,
            lib_names_ar,
            num_kernels_ar,
            num_iters_ar,
            train_ar,
            algo_id,
            td1,
            td2,
        )

        num_clients = len(tids)
        # print(f"[PyScheduler] Num clients is {num_clients}")
        # print(f"[PyScheduler] before starting, profile={profile}, run_eval={run_eval}")

        timings = []

        if run_eval:
            # EVAL path (what you are using for your profiling runs)
            if profile:
                barriers[0].wait()

                # warm-up and setup
                self._sched_lib.schedule(
                    self._scheduler,
                    num_clients,
                    True,      # profile_mode
                    0,         # iter
                    True,      # warmup
                    1,         # warmup_iters
                    algo_id,
                    algo_param,
                    hp_limit,
                    update_start,
                )
                torch.cuda.synchronize()

                # swap to additional kernel files if needed
                for j in range(num_clients):
                    if additional_kernel_files[j] is not None:
                        new_kernel_file = additional_kernel_files[j].encode('utf-8')
                        self._sched_lib.setup_change(
                            self._scheduler,
                            j,
                            new_kernel_file,
                            additional_num_kernels[j],
                            algo_id,   # pass algo_id so C++ knows how to parse CSV
                        )

                print("[PyScheduler] wait after setup_change")
                barriers[0].wait()
                print("[PyScheduler] done!")

                # warmup run
                self._sched_lib.schedule(
                    self._scheduler,
                    num_clients,
                    True,      # profile_mode
                    0,         # iter
                    True,      # warmup
                    10,        # warmup_iters
                    algo_id,
                    algo_param,
                    hp_limit,
                    update_start,
                )
                torch.cuda.synchronize()
                barriers[0].wait()

                # real measurement run
                start = time.time()
                # print("[PyScheduler] call schedule (eval)")
                self._sched_lib.schedule(
                    self._scheduler,
                    num_clients,
                    True,      # profile_mode
                    0,         # iter
                    False,     # warmup
                    0,         # warmup_iters
                    algo_id,
                    algo_param,
                    hp_limit,
                    update_start,
                )
                barriers[0].wait()
                torch.cuda.synchronize()
                print(f"[PyScheduler] Total time is {time.time() - start}")

        else:
            # NON-EVAL path (not really used in your current experiments,
            # but kept consistent with new schedule signature)
            for i in range(num_iters[0]):

                print(f"[PyScheduler] Start iteration {i}")
                if profile:
                    barriers[0].wait()

                    # Needed for backward: swap kernel files after first iter
                    if i == 1:
                        for j in range(num_clients):
                            if additional_kernel_files[j] is not None:
                                new_kernel_file = additional_kernel_files[j].encode('utf-8')
                                self._sched_lib.setup_change(
                                    self._scheduler,
                                    j,
                                    new_kernel_file,
                                    additional_num_kernels[j],
                                    algo_id,  # again, pass algo_id
                                )
                        barriers[0].wait()  # same as your original code

                    start = time.time()
                    print("[PyScheduler] -------------call schedule (train profile)------------")
                    self._sched_lib.schedule(
                        self._scheduler,
                        num_clients,
                        True,      # profile_mode
                        i,         # iter
                        False,     # warmup
                        0,         # warmup_iters
                        algo_id,
                        algo_param,
                        hp_limit,
                        update_start,
                    )
                    torch.cuda.synchronize()

                else:
                    # old per-client schedule_one path (unchanged)
                    start = time.time()
                    for j in range(num_clients):
                        barriers[j].wait()
                        self._sched_lib.schedule_one(self._scheduler, j)
                        torch.cuda.synchronize()

                total_time = time.time() - start
                print(f"[PyScheduler] Iteration {i} took {total_time} sec")
                timings.append(total_time)

            # drop first few warmup iters
            if len(timings) > 3:
                timings = timings[3:]
            print(
                f"[PyScheduler] Avg is {np.median(np.asarray(timings))}, "
                f"Min is {min(timings)} sec"
            )
