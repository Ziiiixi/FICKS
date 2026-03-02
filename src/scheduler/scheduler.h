// scheduler.h
#pragma once

#include <stdio.h>
#include <dlfcn.h>
#include <queue>
#include <vector>
#include <pthread.h>
#include <syscall.h>
#include <pwd.h>
#include <iostream>
#include <string.h>
#include <tuple>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <boost/thread/barrier.hpp>
#include "running_ops_shared.h"

#include "utils_sched.h"

// ---------------- ThreadPool (unchanged) ----------------
class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// (ThreadPool implementation as you already have)

// ---------------- Scheduler ----------------

// Which scheduling algorithm to use
enum class SchedAlgo {
    MULTI_STREAM,  // your "Multiple Streams" baseline
    ORION,
    REEF,
    KRISP
};

class Scheduler {

public:
    boost::barrier sync_barrier;
    
    // Add algo as a member; default to MULTI_STREAM to keep old behavior
    Scheduler(int num_clients, SchedAlgo algo = SchedAlgo::MULTI_STREAM)
        : sync_barrier(num_clients),
          algo_(algo)
    {}

    struct Client {
        int id;
        int cur_iter;
        int seen_k;
        int total_k;   // num_client_kernels[id] (kernels in 1 request)

        // OLD constructor: keep it for compatibility
        // Client(int id_, int cur_iter_, int seen_k_)
        //     : id(id_), cur_iter(cur_iter_), seen_k(seen_k_), total_k(1) {}

        // NEW constructor: used by the PQ for fairness
        Client(int id_, int cur_iter_, int seen_k_, int total_k_)
            : id(id_), cur_iter(cur_iter_), seen_k(seen_k_),
            total_k(total_k_ > 0 ? total_k_ : 1) {}
    };



    // struct RunningOp {
    //     int client_id;
    //     int kernel_id;
    //     op_info* op;              // pointer to the op_info entry
    //     // int tpc_used;             // tpcs used by THIS op at launch
    //     // uint64_t est_finish_ns;   // relative to start_total (same timebase)
    // };



    struct KernelLogEntry {
        int client_id;
        int iter_id;
        int kernel_id;
        int event_id;
        int tpc_used;
        std::string kernel_name;
        std::string model_name;   // <--- ADD THIS
        long long start_ns;
        long long end_ns;
        long long duration_ns;
    };



    struct ClientPriorityLess {
        bool operator()(const Client& a, const Client& b) const {
            // 1) lower iteration gets higher priority
            if (a.cur_iter != b.cur_iter)
                return a.cur_iter > b.cur_iter;

            // 2) compare normalized progress: (seen/total) smaller => higher priority
            //    use cross-multiply to avoid floating point:
            //    a.seen/a.total ? b.seen/b.total
            const long long at = (a.total_k > 0) ? a.total_k : 1;
            const long long bt = (b.total_k > 0) ? b.total_k : 1;

            const long long lhs = (long long)a.seen_k * bt;
            const long long rhs = (long long)b.seen_k * at;

            if (lhs != rhs)
                return lhs > rhs;   // larger fraction => lower priority

            // 3) if same fraction, smaller absolute seen gets slightly higher priority
            if (a.seen_k != b.seen_k)
                return a.seen_k > b.seen_k;

            // 4) if still tie, prefer larger total_k (bigger request) to avoid starvation
            return a.total_k < b.total_k;
        }
    };



    struct CriticalKernel {
        int id;
        int tpc;
    };

    // static std::vector<RunningOp> g_running_ops;

    void profile_prep(std::queue<func_record>** qbuffers,
                      int num_clients,
                      bool reef);
    void profile_reset(int num_clients);

    // SINGLE PUBLIC ENTRY POINT
    void* busy_wait_profile(int num_clients,
                            int iter,
                            bool warmup,
                            int warmup_iters,
                            int algo,      // <--- which algorithm to use
                            bool seq,
                            int depth,
                            int hp_limit,
                            int update_start);

    // Optional: setter if you want to change algo after construction
    void set_algo(SchedAlgo algo) { algo_ = algo; }

private:
    SchedAlgo algo_;

    // Internal implementations for each scheduler
    void* busy_wait_ms(int num_clients,
                       int iter,
                       bool warmup,
                       int warmup_iters,
                       bool seq,
                       int depth,
                       int hp_limit,
                       int update_start);

    void* busy_wait_orion(int num_clients,
                          int iter,
                          bool warmup,
                          int warmup_iters,
                          bool seq,
                          int depth,
                          int hp_limit,
                          int update_start);

    void* busy_wait_reef(int num_clients,
                         int iter,
                         bool warmup,
                         int warmup_iters,
                         bool seq,
                         int depth,
                         int hp_limit,
                         int update_start);

    void* busy_wait_krisp(int num_clients,
                          int iter,
                          bool warmup,
                          int warmup_iters,
                          bool seq,
                          int depth,
                          int hp_limit,
                          int update_start);
    void* busy_wait_ficks(int num_clients,
                          int iter,
                          bool warmup,
                          int warmup_iters,
                          bool seq,
                          int depth,
                          int hp_limit,
                          int update_start);
    void* busy_wait_profile_co(int num_clients,
                        int iter,
                        bool warmup,
                        int warmup_iters,
                        bool seq,
                        int depth,
                        int hp_limit,
                        int update_start);


    void schedule_reef(std::vector<func_record*> frecords,
                       int num_clients,
                       int depth,
                       int hp_client);

};
