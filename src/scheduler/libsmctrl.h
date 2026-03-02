/**
 * Copyright 2023 Joshua Bakita
 * Library to control TPC masks on CUDA launches. Co-opts preexisting debug
 * logic in the CUDA driver library, and thus requires a build with -lcuda.
 */

#include <vector>
#include <stdint.h>
typedef unsigned __int128 uint128_t;
/* PARTITIONING FUNCTIONS */

// Set global default TPC mask for all kernels, incl. CUDA-internal ones
// @param mask   A bitmask of enabled/disabled TPCs (see Notes on Bitmasks)
// Supported: CUDA 10.2, and CUDA 11.0 - CUDA 12.1
extern void libsmctrl_set_global_mask(int idx, int num_clients);
extern void libsmctrl_set_KRISP_mask(int idx, std::vector<int> num_clients);
extern void libsmctrl_set_next_mask();
extern void libsmctrl_set_KRISP_I_mask(uint32_t mask);
extern void libsmctrl_reset_mask();
extern void libsmctrl_set_mask();