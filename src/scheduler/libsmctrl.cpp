#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include "libsmctrl.h"
#include <cuda.h>
#include <string>
#include <stdexcept>
#include <inttypes.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <bitset> 
#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)
static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};

#define LAUNCH_DOMAIN 0x3
#define LAUNCH_PRE_UPLOAD 0x3

static uint64_t g_sm_mask = 0;
static thread_local uint64_t g_next_sm_mask = 0;
static char sm_control_setup_called = 0;

static void launchCallback(void* ukwn, int domain, int cbid, const void* in_params) {
    if (*(uint32_t*)in_params < 0x50) {
        abort(1, 0, "Unsupported CUDA version for callback-based SM masking. Aborting...");
    }

    void* tmd = **((uintptr_t***)in_params + 8);
    if (!tmd) {
        abort(1, 0, "TMD allocation appears NULL; likely forward-compatibility issue.\n");
    }
    uint32_t* lower_ptr = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(tmd) + 84);
    uint32_t* upper_ptr = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(tmd) + 88);

    if (g_next_sm_mask) {
        *lower_ptr = static_cast<uint32_t>(g_next_sm_mask);
        *upper_ptr = static_cast<uint32_t>(g_next_sm_mask >> 32);
        g_next_sm_mask = 0;
    } else if (!*lower_ptr && !*upper_ptr) {
        // printf("yes i am called!!\n  ");
        // printf("%" PRIu64 "\n", g_sm_mask);
        *lower_ptr = static_cast<uint32_t>(g_sm_mask);
        *upper_ptr = static_cast<uint32_t>(g_sm_mask >> 32);
    }
}
static void setup_sm_control_11() {
    using SubscribeFunc = int (*)(uint32_t*, void(*)(void*, int, int, const void*), void*);
    using EnableFunc = int (*)(uint32_t, uint32_t, int, int);
    const void* tbl_base;
    uint32_t my_hndl;
    // Avoid race conditions (setup can only be called once)
    if (__atomic_test_and_set(&sm_control_setup_called, __ATOMIC_SEQ_CST)) {
        return;
    }
    cuGetExportTable(&tbl_base, &callback_funcs_id);

    const uintptr_t* uintptr_tbl_base = static_cast<const uintptr_t*>(tbl_base);
    uintptr_t subscribe_func_addr = *(uintptr_tbl_base + 3);
    uintptr_t enable_func_addr = *(uintptr_tbl_base + 6);
    SubscribeFunc subscribe = reinterpret_cast<SubscribeFunc>(subscribe_func_addr);
    EnableFunc enable = reinterpret_cast<EnableFunc>(enable_func_addr);

    int res = 0;

    res = subscribe(&my_hndl, launchCallback, nullptr);
    // printf("res: %d\n", res);
    if (res) {
        abort(1, 0, "Error subscribi ng to launch callback. CUDA returned error code %d.", res);
    }
    res = enable(1, my_hndl, LAUNCH_DOMAIN, LAUNCH_PRE_UPLOAD);
    // printf("res ENABLE: %d\n", res);
    if (res) {
        abort(1, 0, "Error enabling launch callback. CUDA returned error code %d.", res);
    }

    
}


// Set default mask for all launches
void libsmctrl_set_global_mask(int idx, int num_clients) {
    const int total_tpcs = 1; // Total number of SMs
    const int bits_per_tpc = sizeof(uint64_t) * 8; // Number of bits in uint64_t

    if (num_clients <= 0 || idx < 0 || idx >= num_clients) {
        printf("Invalid client index or number of clients.\n");
        
        return;
    }

    // Calculate the number of SMs per client
    int sm_per_client = total_tpcs / num_clients;
    int start_sm = idx * sm_per_client;
    int end_sm = (idx == num_clients - 1) ? total_tpcs : start_sm + sm_per_client;

    uint64_t mask = 0;

    // Set the bits in the mask for the range of SMs assigned to this client
    for (int i = start_sm; i < end_sm; i++) {
        mask |= (1ULL << i);
    }
    
    // printf("Client %d: start_sm %d end_sm %d \n", idx, start_sm, end_sm);

    mask = ~mask;
    if (!sm_control_setup_called) {
        setup_sm_control_11();
    }
    g_sm_mask = mask;
}


// Set default mask for all launches
void libsmctrl_set_KRISP_mask(int idx, std::vector<int> clients) {
    int total_tpcs = 24; // Total number of SMs
    int num_clients = clients.size();
    // printf("num_clients %d\n", num_clients);
    // Verify the number of clients
    if (num_clients <= 0 || num_clients > total_tpcs) {
        // std::cerr << "Error: Invalid number of clients. Must be between 1 and " << total_tpcs << ".\n";
        printf("Error: Invalid number of clients. Must be between 1 and %d\n", total_tpcs);
        return; // Exit the function or handle the error appropriately
    }

    int sm_per_client = total_tpcs / num_clients; // Determine SMs per client
    int remaining_sm = total_tpcs % num_clients; // Handle remainder

    // Initialize mask to 0
    uint64_t mask = 0;

    auto it = std::find(clients.begin(), clients.end(), idx);
    int position = std::distance(clients.begin(), it);
    int start_sm = position * sm_per_client;
    int end_sm = start_sm + sm_per_client - 1;

    // if (position == num_clients - 1) {
    //     // Last client gets any remaining SMs
    //     end_sm += remaining_sm;
    // }
    // printf("Client %d start_sm %d end_sm %d \n",idx, start_sm, end_sm);
    // Update the mask for this range of SMs
    for (int j = start_sm; j <= end_sm; ++j) {
        mask |= (1 << j); // Set the bit corresponding to this SM
    }
    

 	mask = ~mask;
	if (!sm_control_setup_called) {
        setup_sm_control_11();
	}
	g_sm_mask = mask;
}

void libsmctrl_set_KRISP_I_mask(uint32_t mask){
    if (!sm_control_setup_called) {
        setup_sm_control_11();
    }
    // std::cout << "mask after unset " << std::bitset<32>(mask) << std::endl;
    g_sm_mask = mask;
}

void libsmctrl_set_mask() {
	uint128_t mask = (1ULL << 0);
                    // |(1ULL << 1);
					// |(1ULL << 2);
                    // |(1ULL << 3);
                    // |(1ULL << 4);

                    // |(1ULL << 5);
                    // |(1ULL << 6);
                    // |(1ULL << 7);
                    // |(1ULL << 8);
                    // |(1ULL << 9);


                    // |(1ULL << 10);
                    // |(1ULL << 11);
                    // |(1ULL << 12);
                    // |(1ULL << 13); 
                    // |(1ULL << 14);


                    // |(1ULL << 15);
					// |(1ULL << 16);
                    // |(1ULL << 17);
                    // |(1ULL << 18);
                    // |(1ULL << 19);

					// |(1ULL << 20);
                    // |(1ULL << 21);
                    // |(1ULL << 22);
                    // |(1ULL << 23);
 	mask = ~mask;
	if (!sm_control_setup_called) {
        setup_sm_control_11();
	}
	g_sm_mask = mask;
}


void libsmctrl_reset_mask() {
    uint64_t mask = 0;
    for (int j = 0; j <= 24; ++j) {
        mask |= (1 << j); // Set the bit corresponding to this SM
    }
 	mask = ~mask;
	if (!sm_control_setup_called) {
        setup_sm_control_11();
	}
	g_sm_mask = mask;
}

// Set mask for next launch from this thread
void libsmctrl_set_next_mask() {
	// printf("iam called next_mask\n");
	// uint64_t mask = (1ULL << 1) ;
 	// mask = ~mask;
	if (!sm_control_setup_called) {
		setup_sm_control_11();
	}
	// g_next_sm_mask = mask;
}