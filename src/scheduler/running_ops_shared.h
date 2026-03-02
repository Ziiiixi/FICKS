// running_ops_shared.h
#pragma once
#include <vector>

// forward declare (pointer only)
struct op_info;

// shared struct (NOT nested inside Scheduler)
struct RunningOp {
    int client_id;
    int kernel_id;
    op_info* op;   // pointer only
};

// global shared running-op list
extern std::vector<RunningOp> g_running_ops;
