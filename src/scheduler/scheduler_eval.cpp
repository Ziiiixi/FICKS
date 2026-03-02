#include "scheduler.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip> 
#include <unordered_map>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include "critical_model.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio> 
#define MIN_DURATION 100000 // might need to change this - emperical
using namespace std;
int priority_client = 1;
// globals
void* klib;
vector<vector<op_info>> op_info_vector;
int* fidx;
int* num_client_kernels;
int* num_client_max_iters;
int* num_client_cur_iters;

bool* locked;

std::chrono::time_point<std::chrono::high_resolution_clock>* client_starts;
std::chrono::time_point<std::chrono::high_resolution_clock>* total_client_starts;
bool** client_starts_set;
vector<vector<float>> client_durations;
int num_tpcs = 24;
int max_sms = 48; // v100
queue<struct func_record>** client_buffers;
pthread_mutex_t** client_mutexes;
queue<struct func_record>** buffers;
int* seen;
vector<int> client_progress;
vector<int> func_progress;
// cudnnHandle_t* global_handle0;
// cudnnHandle_t* global_handle1;

// fifo-globals
cudaStream_t sched_stream;
cudaStream_t sync_stream;
cudaEvent_t sched_event;

// profile-globals
cudaStream_t** sched_streams;
cudaStream_t** sync_streams;
cudaEvent_t*** events;
int* streams;
int* event_ids;
int status;
vector<int> max_sms_clients;
vector<bool> is_train;

// reef
int lp_idx = 0;
int penalty = 0;
bool** request_status;
bool* stops;
bool* stop_ack;

//spacial

vector<int> num_cur_clients;
bool *client_finished;
uint32_t mask;
uint32_t *localMask;
uint32_t *localMask_O;

int tpc_usage_count[24] = {0}; 
std::vector<std::string> model_names;
std::vector<bool> is_executing;
// std::vector<Scheduler::RunningOp> g_running_ops;

std::vector<RunningOp> g_running_ops;

// [UT][CO][n] -> multiplier (default 1.0)
static std::vector<std::vector<std::vector<double>>> g_pair;
// [UT][n]     -> multiplier (default 1.0)
static std::vector<std::vector<double>> g_ut;
static bool g_loaded = false;
static float g_TD1 = 1.2f;
static float g_TD2 = 1.1f;
static constexpr int HW_NUM_TPCS = 24;


bool use_excl = false;  // defined elsewhere

void Scheduler::profile_reset(int num_clients) {

	for (int i=0; i<num_clients; i++) {
		seen[i] = 0;
		streams[i] = -1;
		fidx[i] = 0;
	}
}

void Scheduler::profile_prep(queue<func_record>** qbuffers, int num_clients, bool reef) {

	register_functions();
	client_buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		client_buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	int num = num_clients;

	sched_streams = (cudaStream_t**)malloc((num)*sizeof(cudaStream_t*));
	sync_streams = (cudaStream_t**)malloc((num)*sizeof(cudaStream_t*));

	for (int i=0; i<num; i++){
		sched_streams[i] = NULL;
		sync_streams[i] = NULL;
	}

	events = (cudaEvent_t***)malloc((num)*sizeof(cudaEvent_t**));
	for (int i=0; i<num; i++)
		events[i] = NULL;

	create_streams(sched_streams, num, reef);
	create_streams(sync_streams, num, reef);
	create_events(events, num);

	

	seen = (int*)calloc(num,sizeof(int));
	event_ids = (int*)calloc(num, sizeof(int));
	localMask = (uint32_t*)calloc(num,sizeof(uint32_t));
	localMask_O = (uint32_t*)calloc(num,sizeof(uint32_t));

	streams = (int*)malloc(num_clients*sizeof(int));
	for (int i=0; i<num_clients; i++)
		streams[i] = -1;

	sched_stream = 0;

	status = -1;

}



void Scheduler::schedule_reef(vector<func_record*> frecords, int num_clients, int depth, int hp_client) {

	// schedule based on REEF policy
    
	// if (num_clients==1) {
	// 	if (frecords[0] != NULL) {
	// 		schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0][event_ids[0]], seen, event_ids, 0);
	// 		pop_from_queue(client_buffers[0], client_mutexes[0], 0);
	// 	}
	// 	return;
	// }
    
	// check for malloc operations
	for (int i=0; i<num_clients; i++) {
		if (frecords[i] != NULL) {
            if (frecords[i]->type == MALLOC_RECORD ||
                frecords[i]->type == MEMCPY_RECORD || 
                frecords[i]->type == MEMSET_RECORD ||
                frecords[i]->type == FREE_RECORD){
				schedule_kernel(*(frecords[i]), sched_streams[i], i, events[i][event_ids[i]], seen, event_ids, i);
				pop_from_queue(client_buffers[i], client_mutexes[i], i);
				return;
			}
		}
	}

    bool canSchedule[num_clients];
    for (int i = 0; i < num_clients; ++i) {
        canSchedule[i] = true;
        if (event_ids[i] >= 1) {
            if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                // printf("kernel %d finished\n", event_ids[i]);
                unsetmask(i);
            }
            else{
                canSchedule[i] = false; 
            }
        }
    }

	// if hp is found, schedule
	if (frecords[hp_client] != NULL && canSchedule[hp_client]) {
		int hp_idx = seen[hp_client];
        op_info op_info_1 = op_info_vector[hp_client][hp_idx];
        int tpc_usage = op_info_1.sm_used / 2;
        tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;

        if(num_tpcs >= tpc_usage){
            setmask(tpc_usage, hp_client);
            schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
            pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
        }
	}
	else {
		for (int i=0; i<hp_client; i++) {
			if (frecords[i] != NULL)
				penalty += 1;
		}
		if (penalty>=depth) {
			// schedule all
			for (int i=0; i<hp_client; i++) {
				if (frecords[i] != NULL && canSchedule[i]) {
                    op_info op_info_0 = op_info_vector[i][seen[i]];
                    int tpc_usage = op_info_0.sm_used / 2;
                    tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;
                    if(num_tpcs > 0){
                        setmask(min(num_tpcs, tpc_usage), i);
                        schedule_kernel(*(frecords[i]), sched_streams[i], i, events[i][event_ids[i]], seen, event_ids, i);
                        pop_from_queue(client_buffers[i], client_mutexes[i], i);
                    }
				}
			}
			penalty = 0;
		}
	}
}

static inline bool is_admin_record(const func_record& r) {
    return (r.type == MALLOC_RECORD ||
            r.type == MEMCPY_RECORD ||
            r.type == MEMSET_RECORD ||
            r.type == FREE_RECORD);
}



// original orion
// void* Scheduler::busy_wait_orion(int num_clients,
//                                  int iter,
//                                  bool warmup,
//                                  int warmup_iters,
//                                  bool seq,
//                                  int depth,
//                                  int hp_limit,
//                                  int update_start)
// {
//     printf("Entered busy_wait_orion! Num clients = %d\n", num_clients);

//     auto start_total = std::chrono::high_resolution_clock::now();
//     std::vector<bool> total_client_set(num_clients, false);

//     // ------------------------------------------------------------
//     // Round-robin HP pointer (persists across calls)
//     // Start it from priority_client if valid, otherwise 0.
//     // ------------------------------------------------------------
//     static int rr_next_hp = -1;
//     if (rr_next_hp < 0) {
//         if (priority_client >= 0 && priority_client < num_clients) rr_next_hp = priority_client;
//         else rr_next_hp = 0;
//     }

//     while (true) {
//         std::vector<func_record*> frecords(num_clients, nullptr);

//         // 1) Pull one record per client, set start timestamps
//         for (int i = 0; i < num_clients; ++i) {
//             if (is_executing[i]) continue;
//             if (seen[i] == num_client_kernels[i]) continue;

//             pthread_mutex_lock(client_mutexes[i]);
//             volatile int sz = client_buffers[i]->size();
//             if (sz > 0) {
//                 frecords[i] = &client_buffers[i]->front();
//                 int cur_iter = num_client_cur_iters[i];
//                 if (seen[i] == 0 && !client_starts_set[i][cur_iter]) {
//                     client_starts[i] = std::chrono::high_resolution_clock::now();
//                     client_starts_set[i][cur_iter] = true;
//                     if (!total_client_set[i]) {
//                         total_client_starts[i] = std::chrono::high_resolution_clock::now();
//                         total_client_set[i] = true;
//                     }
//                 }
//             }
//             pthread_mutex_unlock(client_mutexes[i]);
//         }

//         // 2) Check events & build canSchedule[]
//         std::vector<bool> canSchedule(num_clients, true);
//         for (int i = 0; i < num_clients; ++i) {
//             if (event_ids[i] >= 1) {
//                 if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
//                     unsetmask(i);
//                 } else {
//                     canSchedule[i] = false;
//                 }
//             }
//         }

//         // 3) ORION scheduling step
//         // ----------------------------------------------------
//         // 3.1 Admin ops: always schedule, no mask
//         for (int i = 0; i < num_clients; ++i) {
//             func_record* rec = frecords[i];
//             if (!rec) continue;
//             if (!is_admin_record(*rec)) continue;

//             schedule_kernel(*rec, sched_streams[i], i,
//                             events[i][event_ids[i]], seen, event_ids, i);
//             pop_from_queue(client_buffers[i], client_mutexes[i], i);
//             frecords[i] = nullptr;  // consumed
//         }

//         // ----------------------------------------------------
//         // 3.2 HP client: STRICT ROUND ROBIN among eligible clients
//         // ----------------------------------------------------
//         int hp_client = -1;

//         // pick first eligible client starting from rr_next_hp
//         for (int k = 0; k < num_clients; ++k) {
//             int cand = (rr_next_hp + k) % num_clients;
//             func_record* rec = frecords[cand];
//             if (!rec) continue;
//             // if (!canSchedule[cand]) continue;
//             if (is_admin_record(*rec)) continue;  // admin already handled
//             hp_client = cand;
//             break;
//         }

//         long long hp_endT = 0;      // approximate HP completion window
//         int hp_tpcs_used  = 0;      // track how many TPCs HP uses
//         long long hp_op_duration = 0;

//         if (hp_client >= 0) {
//             // snapshot HP op info BEFORE scheduling (seen[] might change after schedule_kernel)
//             op_info op_hp = op_info_vector[hp_client][seen[hp_client]];

//             int tpc_usage = op_hp.sm_used / 2;
//             if (tpc_usage < 1) tpc_usage = 1;
//             if (tpc_usage > num_tpcs) tpc_usage = num_tpcs;

//             hp_tpcs_used = tpc_usage;
//             hp_op_duration = (long long)op_hp.duration;

//             if (tpc_usage > 0) {
//                 // setmask(tpc_usage, hp_client);

//                 schedule_kernel(*frecords[hp_client],
//                                 sched_streams[hp_client],
//                                 hp_client,
//                                 events[hp_client][event_ids[hp_client]],
//                                 seen, event_ids, hp_client);

//                 pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
//                 frecords[hp_client] = nullptr;

//                 // approximate HP finish time using threshold td1
//                 double td1 = 0.8;  // tune if you like
//                 auto now = std::chrono::high_resolution_clock::now();
//                 long long now_ns =
//                     std::chrono::duration_cast<std::chrono::nanoseconds>(
//                         now.time_since_epoch()).count();
//                 hp_endT = now_ns + (long long)(td1 * (double)hp_op_duration);

//                 // advance RR pointer ONLY when HP actually launches
//                 rr_next_hp = (hp_client + 1) % num_clients;
//             } else {
//                 // HP candidate invalid -> treat as no HP this round
//                 hp_client = -1;
//             }
//         }

//         // 3.3 BE clients: co-run with HP if there are TPCs left
//         if (hp_endT > 0) {
//             int remaining_tpcs = num_tpcs - hp_tpcs_used;
//             if (remaining_tpcs < 0) remaining_tpcs = 0;

//             for (int i = 0; i < num_clients; ++i) {
//                 if (i == hp_client) continue;
//                 func_record* rec = frecords[i];
//                 if (!rec || !canSchedule[i]) continue;
//                 if (is_admin_record(*rec)) continue;

//                 op_info op_be = op_info_vector[i][seen[i]];

//                 // simple td2 test: if BE is not too long vs HP, co-run
//                 double td2 = 2.0;  // tune if you like
//                 if (hp_op_duration > 0 &&
//                     (double)op_be.duration > td2 * (double)hp_op_duration)
//                 {
//                     continue;
//                 }

//                 int tpc_usage = op_be.sm_used / 2;
//                 if (tpc_usage < 1) tpc_usage = 1;
//                 if (tpc_usage > remaining_tpcs) tpc_usage = remaining_tpcs;
//                 if (tpc_usage <= 0) continue;

//                 // setmask(tpc_usage, i);

//                 schedule_kernel(*rec,
//                                 sched_streams[i],
//                                 i,
//                                 events[i][event_ids[i]],
//                                 seen, event_ids, i);

//                 pop_from_queue(client_buffers[i], client_mutexes[i], i);
//                 frecords[i] = nullptr;

//                 remaining_tpcs -= tpc_usage;
//                 if (remaining_tpcs <= 0) break;
//             }
//         } else {
//             // HP not launched this round: let BE run if they can
//             for (int i = 0; i < num_clients; ++i) {
//                 func_record* rec = frecords[i];
//                 // if (!rec || !canSchedule[i]) continue;
//                 if (!rec) continue;
//                 if (is_admin_record(*rec)) continue;

//                 op_info op_be = op_info_vector[i][seen[i]];
//                 int tpc_usage = op_be.sm_used / 2;
//                 if (tpc_usage < 1) tpc_usage = 1;
//                 if (tpc_usage > num_tpcs) tpc_usage = num_tpcs;
//                 if (tpc_usage <= 0) continue;

//                 // setmask(tpc_usage, i);

//                 schedule_kernel(*rec,
//                                 sched_streams[i],
//                                 i,
//                                 events[i][event_ids[i]],
//                                 seen, event_ids, i);

//                 pop_from_queue(client_buffers[i], client_mutexes[i], i);
//                 frecords[i] = nullptr;
//             }
//         }

//         // 4) iteration completion logic
//         int finished = 0;
//         for (int i = 0; i < num_clients; ++i) {

//             if (   num_client_cur_iters[i] == num_client_max_iters[i]
//                 || (warmup && num_client_cur_iters[i] == warmup_iters)
//                 || stop_ack[i])
//             {
//                 finished += 1;
//                 continue;
//             }

//             if (seen[i] == num_client_kernels[i]) {
//                 if (!locked[i]) {
//                     pthread_mutex_lock(client_mutexes[i]);
//                     locked[i] = true;
//                     DEBUG_PRINT("LOCK CLIENT %d\n", i);
//                 }

//                 bool ready = true;
//                 if (seq) {
//                     if (event_ids[0] >= 1 &&
//                         cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess)
//                     {
//                         ready = false;
//                     }
//                 } else {
//                     if (event_ids[i] >= 1 &&
//                         cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess)
//                     {
//                         ready = false;
//                     }
//                 }

//                 if (ready) {
//                     // unsetmask(i);

//                     seen[i] = 0;
//                     if (seq) event_ids[0] = 0;
//                     event_ids[i] = 0;
//                     streams[i] = -1;
//                     fidx[i]    = 0;
//                     request_status[i][num_client_cur_iters[i]] = true;
//                     pthread_mutex_unlock(client_mutexes[i]);
//                     num_client_cur_iters[i] += 1;
//                     locked[i] = false;
//                     client_progress[i] = 0;

//                     auto end = std::chrono::high_resolution_clock::now();
//                     float duration_ms =
//                         std::chrono::duration_cast<std::chrono::microseconds>(
//                             end - client_starts[i]).count() / 1000.0f;
//                     client_durations[i].push_back(duration_ms);
//                 }
//             }
//         }

//         if (finished == num_clients) break;
//     }

//     if (!warmup) {
//         auto end_total = std::chrono::high_resolution_clock::now();
//         long long duration_ns =
//             std::chrono::duration_cast<std::chrono::nanoseconds>(
//                 end_total - start_total).count();
//         printf("Total ORION loop took %lld ns\n", duration_ns);
//     }

//     return NULL;
// }




void* Scheduler::busy_wait_orion(int num_clients,
                                 int iter,
                                 bool warmup,
                                 int warmup_iters,
                                 bool seq,
                                 int depth,
                                 int hp_limit,
                                 int update_start)
{
	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
	int start0 = 0;
	int start1 = 0;

	int prev_large = -1;
	int hp_running = -1;

	bool inf_finished = false;
	bool started = false;
 	std::chrono::time_point<std::chrono::system_clock> start_time;
	auto start_total = std::chrono::high_resolution_clock::now();

	vector<bool> total_client_set(num_clients, false);
	vector<int> profiles(num_clients, -1);
	vector<int> cur_sms(num_clients, -1);
	int hp_client = num_clients - 1;

	// NEW: RR pointer for HP client
	static int rr_next_hp = -1;
	if (rr_next_hp < 0 || rr_next_hp >= num_clients) {
		rr_next_hp = (num_clients > 0) ? (num_clients - 1) : 0;
	}

	bool large_found = false;
	long sum = 0; // sum of durations of ongoing BE kernels
	long size = 0; // sum of sizes of in-the-queues BE kernels
	int start = -1;

	// BS - works only for 2 clients for now
	// TODO: check this
	int low_sms = 0;
	int high_sms = max_sms_clients[0]; // 0 is the lp client
	int sm_threshold = max_sms_clients[0]/2;
	float hp_iter_duration = 0.0; // hp client rolling duration
	float hp_limit_float = (float)hp_limit;

	(void)iter;
	(void)start0;
	(void)start1;
	(void)prev_large;
	(void)hp_running;
	(void)inf_finished;
	(void)started;
	(void)start_time;

	auto client_done = [&](int i) -> bool {
		return (
			(num_client_cur_iters[i] == num_client_max_iters[i]) ||
			(warmup && (num_client_cur_iters[i] == warmup_iters)) ||
			(stop_ack[i] == true)
		);
	};

	while(1) {
		vector<func_record*> frecords(num_clients, NULL);
		vector<char> scheduled_once(num_clients, 0); // NEW: enforce <=1 schedule per client per loop
		size = 0;

		for (int i = 0; i < num_clients; i++) {
			if (seen[i] == num_client_kernels[i])
				continue;

			pthread_mutex_lock(client_mutexes[i]);
			volatile int sz = client_buffers[i]->size();
			if (sz > 0) {
				frecords[i] = &(client_buffers[i]->front());
				int cur_iter = num_client_cur_iters[i];
				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					client_starts[i] = std::chrono::high_resolution_clock::now();
					client_starts_set[i][cur_iter] = true;
					if (!total_client_set[i]) {
						total_client_starts[i] = std::chrono::high_resolution_clock::now();
						total_client_set[i] = true;
					}
				}
			}
			pthread_mutex_unlock(client_mutexes[i]);
		}

		// =========================================================
		// NEW: pick HP client in RR order (among not-done clients)
		// =========================================================
		int hp_this_round = -1;
		for (int off = 0; off < num_clients; ++off) {
			int cand = (rr_next_hp + off) % num_clients;
			if (client_done(cand)) continue;
			hp_this_round = cand;
			break;
		}
		if (hp_this_round >= 0) {
			hp_client = hp_this_round;
			rr_next_hp = (hp_client + 1) % num_clients; // rotate every while-loop
		}

		// HP-specific local threshold/update behavior (per-round)
		int local_sm_threshold = sm_threshold;
		int local_update_start = update_start;
		if (hp_client >= 0 && !is_train[hp_client]) {
			local_sm_threshold = max_sms;
			local_update_start = INT_MAX;
		}

		// -------------------------
		// 1) Schedule HP (at most once)
		// -------------------------
		if (hp_client >= 0 && frecords[hp_client] != NULL && !scheduled_once[hp_client]) {
			op_info op_info_1 = op_info_vector[hp_client][seen[hp_client]];
			schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client,
			                events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
			streams[hp_client] = 1;
			profiles[hp_client] = op_info_1.profile;
			cur_sms[hp_client] = op_info_1.sm_used;

			status = 1;
			pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
			scheduled_once[hp_client] = 1;
		}

		// -------------------------
		// 2) Schedule BE in RR (each client at most once)
		// -------------------------
		int anchor = (start >= 0) ? start : hp_client;  // keep BE RR feel close to original
		if (anchor < 0) anchor = -1;

		int end = anchor + num_clients; // scan one full ring
		for (int t = anchor + 1; t < end; t++) {
			int j = (num_clients > 0) ? (t % num_clients) : 0;

			if (j == hp_client) continue;
			if (scheduled_once[j]) continue; // NEW: strictly once per loop

			if (frecords[j] != NULL) { // low priority
				op_info op_info_0 = op_info_vector[j][seen[j]];
				bool schedule = false;

				bool is_admin =
					(frecords[j]->type == MALLOC_RECORD) ||
					(frecords[j]->type == MEMCPY_RECORD) ||
					(frecords[j]->type == MEMSET_RECORD) ||
					(frecords[j]->type == FREE_RECORD);

				if ((num_clients == 1) || (hp_client < 0) || (seen[hp_client] == 0) || is_admin)
					schedule = true;
				else if (num_client_cur_iters[j] <= 10 || num_client_cur_iters[hp_client] >= num_client_max_iters[hp_client]) {
					schedule = true;
				}
				else if (seen[hp_client] >= local_update_start &&
				         (op_info_0.sm_used <= local_sm_threshold &&
				          cudaEventQuery(*(events[hp_client][local_update_start - 1])) == cudaSuccess))
					schedule = true;
				else if (seen[hp_client] > 0 &&
				         (size + op_info_0.sm_used <= local_sm_threshold) &&
				         ((op_info_0.profile == -1 || profiles[hp_client] == -1 ||
				           (profiles[hp_client] != op_info_0.profile))))
					schedule = true;

				if (schedule && large_found) {
					bool do_schedule = true;
					for (int k = 0; k < num_clients; k++) { // NEW: all non-HP clients
						if (k == hp_client) continue;
						if (event_ids[k] >= 1) {
							cudaError_t q = cudaEventQuery(*(events[k][event_ids[k]-1]));
							if (q != cudaSuccess) {
								do_schedule = false;
								break;
							}
						}
					}
					if (do_schedule) {
						large_found = false;
						sum = 0;
					}
					else {
						schedule = false;
					}
				}

				if (schedule) {
					size += op_info_0.sm_used;
					if (!is_admin)
						sum += op_info_0.duration;

					if (hp_client >= 0 &&
					    sum > depth &&
					    num_client_cur_iters[hp_client] < num_client_max_iters[hp_client]) {
						large_found = true;
					}

					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
					status = 0;
					pop_from_queue(client_buffers[j], client_mutexes[j], j);

					streams[j] = 0;
					start = j;
					scheduled_once[j] = 1; // NEW
				}
			}
		}

		int finished = 0;
		for (int i = 0; i < num_clients; i++) {
			if (
				(num_client_cur_iters[i] == num_client_max_iters[i])
				|| (warmup && (num_client_cur_iters[i] == warmup_iters))
				|| (stop_ack[i] == true)
			)
				finished += 1;
			else if (seen[i] == num_client_kernels[i]) {
				// check if GPU work for this client has finished
				if (!locked[i]) {
					pthread_mutex_lock(client_mutexes[i]);
					locked[i] = true;
					DEBUG_PRINT("LOCK CLIENT %d\n", i);
				}
				bool ready = true;
				if (seq) {
					if (event_ids[0] >= 1) {
						cudaError_t status = cudaEventQuery(*(events[0][event_ids[0]-1]));
						if (status != cudaSuccess)
							ready &= false;
					}
				}
				else {
					if (event_ids[i] >= 1) {
						cudaError_t status = cudaEventQuery(*(events[i][event_ids[i]-1]));
						if (status != cudaSuccess)
							ready &= false;
					}
				}
                if (ready) {

                    seen[i] = 0;
                    if (seq) event_ids[0] = 0;
                    event_ids[i] = 0;
                    streams[i] = -1;
                    fidx[i]    = 0;
                    request_status[i][num_client_cur_iters[i]] = true;
                    pthread_mutex_unlock(client_mutexes[i]);
                    num_client_cur_iters[i] += 1;
                    locked[i] = false;
                    client_progress[i] = 0;

                    auto end = std::chrono::high_resolution_clock::now();
                    float duration_ms =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            end - client_starts[i]).count() / 1000.0f;
                    client_durations[i].push_back(duration_ms);

                    // keep your hp tuning logic; hp now rotates
                    if (!seq && i == hp_client && is_train[hp_client]) {
                        printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration_ms);
                        hp_iter_duration += duration_ms;
                        if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
                            float hp_avg_duration = hp_iter_duration / 10.0f;
                            printf("--------------------- Average iter duration for client %d is %f ms, limit is %f ms, sm_threshold is %d\n",
                                   i, hp_avg_duration, hp_limit_float, sm_threshold);
                            hp_iter_duration = 0.0f;

                            if (hp_avg_duration > hp_limit_float) {
                                high_sms = sm_threshold;
                                sm_threshold = (low_sms + high_sms) / 2;
                            } else {
                                low_sms = sm_threshold;
                                sm_threshold = (low_sms + high_sms) / 2;
                            }
                        }
                    }
                }
            }
        }

        if (finished == num_clients) break;
    }

    if (!warmup) {
        auto end_total = std::chrono::high_resolution_clock::now();
        long long duration_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_total - start_total).count();
        printf("Total ORION loop took %lld ns\n", duration_ns);
    }

    return NULL;
}

// void* Scheduler::busy_wait_orion(int num_clients,
//                                  int iter,
//                                  bool warmup,
//                                  int warmup_iters,
//                                  bool seq,
//                                  int depth,
//                                  int hp_limit,
//                                  int update_start)
// {
// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	int hp_client = num_clients-1;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

// 	// if hp is inference, use max_sms + also there is no update phase
// 	if (!is_train[hp_client]) {
// 		sm_threshold = max_sms;
// 		update_start = INT_MAX;
// 	}

// 	while(1) {
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {
// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 				//if (seen[i] == num_client_kernels[i]-1)
// 				//	continue;
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


//         if (frecords[hp_client] != NULL) { // high priority

//             op_info op_info_1 = op_info_vector[hp_client][seen[hp_client]];
//             schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
//             streams[hp_client] = 1;
//             profiles[hp_client] = op_info_1.profile;
//             cur_sms[hp_client] = op_info_1.sm_used;

//             status = 1;
//             pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
//         }
//         //start = -1;
//         int end = start + num_clients; // start+1+num_clients-1
//         for (int t=start+1; t<end; t++) {
//             // Do round-robin for the BE clients
//             int j = t % (num_clients-1);
//             if (frecords[j] != NULL) { // low priority
//                 op_info op_info_0 = op_info_vector[j][seen[j]];
//                 bool schedule = false;

//                 //printf("%d, %d, %d\n", low_sms, high_sms, sm_threshold);

//                 if ((num_clients==1) || (seen[hp_client]==0) || (frecords[j]->type == MALLOC_RECORD) || (frecords[j]->type == MEMCPY_RECORD) || (frecords[j]->type == MEMSET_RECORD) || (frecords[j]->type == FREE_RECORD))
//                     schedule = true;
//                 else if (num_client_cur_iters[j] <= 10 || num_client_cur_iters[hp_client] >= num_client_max_iters[hp_client]) {
//                     schedule = true;
//                 }
//                 else if (seen[hp_client] >= update_start && (op_info_0.sm_used <= sm_threshold && cudaEventQuery(*(events[hp_client][update_start-1])) == cudaSuccess)) // && (op_info_0.sm_used <= 10*sm_threshold))
//                     schedule = true;
//                 else if (seen[hp_client]>0 && (size + op_info_0.sm_used <= sm_threshold) &&  ((op_info_0.profile == -1 || profiles[hp_client]==-1 || (profiles[hp_client] != op_info_0.profile))))
//                     schedule = true;
//                 if (schedule && large_found) {
//                     bool do_schedule = true;
//                     for (int k=0; k<num_clients-1; k++) {
//                         if (event_ids[k]>=5) {
//                             cudaError_t status = cudaEventQuery(*(events[k][event_ids[k]-1]));
//                             if (status != cudaSuccess) {
//                                 do_schedule = false;
//                                 break;
//                             }
//                         }
//                     }
//                     if (do_schedule) {
//                         large_found = false;
//                         sum = 0;
//                     }
//                     else
//                         schedule = false;
//                 }
//                 if (schedule) {
//                     //if (op_info_0.duration > depth && num_client_cur_iters[1] < num_client_max_iters[1] && seen[1]==0) {
//                         //block = true;
//                     size += op_info_0.sm_used;
//                     if ((frecords[j]->type != MALLOC_RECORD) && (frecords[j]->type != MEMCPY_RECORD) && (frecords[j]->type != MEMSET_RECORD) && (frecords[j]->type != FREE_RECORD))
//                         sum += op_info_0.duration;
//                     if (sum > depth && num_client_cur_iters[hp_client] < num_client_max_iters[hp_client]) {
//                         large_found = true;
//                     }
//                     schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
//                     status = 0;
//                     pop_from_queue(client_buffers[j], client_mutexes[j], j);

//                     streams[j] = 0;
//                     start = j;
//                 }
//             }
//         }
		

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {
// 			//printf("Client %d, seen is %d, all is %d\n", i, seen[i], num_client_kernels[i]);
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				finished += 1;
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						cudaError_t status = cudaEventQuery(*(events[0][event_ids[0]-1]));
// 						if (status != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						cudaError_t status = cudaEventQuery(*(events[i][event_ids[i]-1]));
// 						if (status != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					DEBUG_PRINT("UNLOCK CLIENT %d\n", i);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;

// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 					if (!seq && i==hp_client && is_train[hp_client]) {
// 						printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration);
// 						hp_iter_duration += duration;
// 						if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
// 							float hp_avg_duration = hp_iter_duration/10.0;
// 							printf("--------------------- Average iter duration for client 1 is %f ms, limit is %f ms, sm_threshold is %d\n", hp_avg_duration, hp_limit_float, sm_threshold);
// 							hp_iter_duration = 0;

// 							// TODO: add better stopping conditions
// 							if (hp_avg_duration > hp_limit_float) {
// 								high_sms = sm_threshold;
// 								sm_threshold = (low_sms+high_sms)/2;
// 							}
// 							else {
// 								low_sms = sm_threshold;
// 								sm_threshold = (low_sms+high_sms)/2;
// 							}
// 						}
// 					}
// 					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
// 				}
// 				if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				) {
// 					finished += 1;
// 					if (!warmup) {
// 						auto end_total = std::chrono::high_resolution_clock::now();
// 						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
// 						duration /= 1000.0;
// 						printf("Client %d, Total loop took %f sec\n", i, duration);
// 						if (i==num_clients-1) {
// 							for (int k=0; k<num_clients-1; k++) {
// 								printf("======= Client %d has done %d iterations\n", k, num_client_cur_iters[k]);
// 								if (!locked[k])
// 									pthread_mutex_lock(client_mutexes[k]);
// 								stops[k] = true;
// 								if (!locked[k])
// 									pthread_mutex_unlock(client_mutexes[k]);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}

// 		if (finished==num_clients) {
// 			printf("EXIT LOOP!\n");
// 			break;
// 		}

// 	}
// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

// void* Scheduler::busy_wait_orion(int num_clients,
//                                  int iter,
//                                  bool warmup,
//                                  int warmup_iters,
//                                  bool seq,
//                                  int depth,
//                                  int hp_limit,
//                                  int update_start)
// {
// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
// 	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	int hp_client = num_clients - 1;

// 	bool large_found = false;
// 	long sum = 0;  // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0] / 2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

// 	// if hp is inference, use max_sms + also there is no update phase
// 	if (!is_train[hp_client]) {
// 		sm_threshold = max_sms;
// 		update_start = INT_MAX;
// 	}

// 	// -------- debug helpers (local lambdas) --------
// 	auto rec_type_str = [&](int type) -> const char* {
// 		switch (type) {
// 			case MALLOC_RECORD: return "MALLOC";
// 			case MEMCPY_RECORD: return "MEMCPY";
// 			case MEMSET_RECORD: return "MEMSET";
// 			case FREE_RECORD:   return "FREE";
// 			default:            return "KERNEL";
// 		}
// 	};

// 	auto print_schedule_msg = [&](int cid, const char* who, const char* reason, const op_info& opi, func_record* fr) {
// 		printf("[ORION][SCHED][%s] client=%d seen=%d iter=%d type=%s "
// 		       "sm_used=%d profile=%d dur=%ld | reason=%s | size=%ld sum=%ld large_found=%d sm_th=%d hp_seen=%d hp_iter=%d\n",
// 		       who,
// 		       cid,
// 		       seen[cid],
// 		       num_client_cur_iters[cid],
// 		       (fr ? rec_type_str(fr->type) : "NULL"),
// 		       opi.sm_used,
// 		       opi.profile,
// 		       (long)opi.duration,
// 		       reason,
// 		       size,
// 		       sum,
// 		       (int)large_found,
// 		       sm_threshold,
// 		       seen[hp_client],
// 		       num_client_cur_iters[hp_client]);
// 	};

// 	auto print_nosched_msg = [&](int cid, const char* reason, const op_info& opi, func_record* fr) {
// 		printf("[ORION][SKIP] client=%d seen=%d iter=%d type=%s "
// 		       "sm_used=%d profile=%d dur=%ld | reason=%s | size=%ld sum=%ld large_found=%d sm_th=%d hp_seen=%d hp_iter=%d hp_profile=%d\n",
// 		       cid,
// 		       seen[cid],
// 		       num_client_cur_iters[cid],
// 		       (fr ? rec_type_str(fr->type) : "NULL"),
// 		       opi.sm_used,
// 		       opi.profile,
// 		       (long)opi.duration,
// 		       reason,
// 		       size,
// 		       sum,
// 		       (int)large_found,
// 		       sm_threshold,
// 		       seen[hp_client],
// 		       num_client_cur_iters[hp_client],
// 		       profiles[hp_client]);
// 	};

// 	// Dedup skip logs: only print once for same (client, kernel, iter, reason)
// 	vector<int> last_skip_seen(num_clients, -1);
// 	vector<int> last_skip_iter(num_clients, -1);
// 	vector<std::string> last_skip_reason(num_clients, "");

// 	auto clear_skip_cache_for_client = [&](int cid) {
// 		last_skip_seen[cid] = -1;
// 		last_skip_iter[cid] = -1;
// 		last_skip_reason[cid].clear();
// 	};

// 	auto print_nosched_msg_once = [&](int cid, const char* reason, const op_info& opi, func_record* fr) {
// 		if (last_skip_seen[cid] == seen[cid] &&
// 		    last_skip_iter[cid] == num_client_cur_iters[cid] &&
// 		    last_skip_reason[cid] == std::string(reason)) {
// 			return;
// 		}

// 		last_skip_seen[cid] = seen[cid];
// 		last_skip_iter[cid] = num_client_cur_iters[cid];
// 		last_skip_reason[cid] = reason;

// 		print_nosched_msg(cid, reason, opi, fr);
// 	};

// 	while (1) {
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i = 0; i < num_clients; i++) {
// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 				//if (seen[i] == num_client_kernels[i]-1)
// 				//	continue;
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}

// 		if (frecords[hp_client] != NULL) { // high priority
// 			op_info op_info_1 = op_info_vector[hp_client][seen[hp_client]];
// 			print_schedule_msg(hp_client, "HP", "HP client always scheduled when queue non-empty", op_info_1, frecords[hp_client]);

// 			schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client,
// 			                events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
// 			streams[hp_client] = 1;
// 			profiles[hp_client] = op_info_1.profile;
// 			cur_sms[hp_client] = op_info_1.sm_used;

// 			status = 1;
// 			pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);

// 			// HP scheduled a new kernel, clear old skip cache for HP (mostly unused, but keeps state clean)
// 			clear_skip_cache_for_client(hp_client);
// 		}

// 		//start = -1;
// 		int end = start + num_clients; // start+1+num_clients-1
// 		for (int t = start + 1; t < end; t++) {
// 			// Do round-robin for the BE clients
// 			int j = t % (num_clients - 1);
// 			if (frecords[j] != NULL) { // low priority
// 				op_info op_info_0 = op_info_vector[j][seen[j]];
// 				bool schedule = false;
// 				const char* schedule_reason = nullptr;

// 				bool is_mem_like =
// 					(frecords[j]->type == MALLOC_RECORD) ||
// 					(frecords[j]->type == MEMCPY_RECORD) ||
// 					(frecords[j]->type == MEMSET_RECORD) ||
// 					(frecords[j]->type == FREE_RECORD);

// 				if ((num_clients == 1) || (seen[hp_client] == 0) || is_mem_like) {
// 					schedule = true;
// 					if (num_clients == 1) schedule_reason = "single client";
// 					else if (seen[hp_client] == 0) schedule_reason = "HP has not started yet (seen[hp]==0)";
// 					else schedule_reason = "memory/alloc/free record bypass";
// 				}
// 				else if (num_client_cur_iters[j] <= 10 ||
// 				         num_client_cur_iters[hp_client] >= num_client_max_iters[hp_client]) {
// 					schedule = true;
// 					if (num_client_cur_iters[j] <= 10)
// 						schedule_reason = "BE warm iterations (num_client_cur_iters[j] <= 10)";
// 					else
// 						schedule_reason = "HP finished max iterations";
// 				}
// 				else if (seen[hp_client] >= update_start &&
// 				         (op_info_0.sm_used <= sm_threshold &&
// 				          cudaEventQuery(*(events[hp_client][update_start - 1])) == cudaSuccess)) {
// 					schedule = true;
// 					schedule_reason = "HP update phase reached and update_start-1 event completed and BE sm <= threshold";
// 				}
// 				else if (seen[hp_client] > 0 &&
// 				         (size + op_info_0.sm_used <= sm_threshold) &&
// 				         ((op_info_0.profile == -1 || profiles[hp_client] == -1 ||
// 				          (profiles[hp_client] != op_info_0.profile)))) {
// 					schedule = true;
// 					schedule_reason = "fits SM threshold and profile compatible/different from HP";
// 				}

// 				if (schedule && large_found) {
// 					bool do_schedule = true;
// 					int blocking_client = -1;
// 					for (int k = 0; k < num_clients - 1; k++) {
// 						if (event_ids[k] >= 5) {
// 							cudaError_t ev_status = cudaEventQuery(*(events[k][event_ids[k] - 1]));
// 							if (ev_status != cudaSuccess) {
// 								do_schedule = false;
// 								blocking_client = k;
// 								break;
// 							}
// 						}
// 					}
// 					if (do_schedule) {
// 						printf("[ORION][LARGE_FOUND] all BE outstanding events completed, clearing large_found (sum reset)\n");
// 						large_found = false;
// 						sum = 0;
// 					}
// 					else {
// 						print_nosched_msg_once(j, "large_found gate active: waiting previous BE event completion", op_info_0, frecords[j]);
// 						if (blocking_client >= 0) {
// 							printf("[ORION][SKIP-DETAIL] client=%d blocked by BE client=%d unfinished event_id=%d\n",
// 							       j, blocking_client, event_ids[blocking_client] - 1);
// 						}
// 						schedule = false;
// 					}
// 				}

// 				if (schedule) {
// 					print_schedule_msg(j, "BE", (schedule_reason ? schedule_reason : "unknown"), op_info_0, frecords[j]);

// 					size += op_info_0.sm_used;
// 					if ((frecords[j]->type != MALLOC_RECORD) &&
// 					    (frecords[j]->type != MEMCPY_RECORD) &&
// 					    (frecords[j]->type != MEMSET_RECORD) &&
// 					    (frecords[j]->type != FREE_RECORD))
// 						sum += op_info_0.duration;

// 					if (sum > depth && num_client_cur_iters[hp_client] < num_client_max_iters[hp_client]) {
// 						large_found = true;
// 						printf("[ORION][LARGE_FOUND] client=%d caused sum=%ld > depth=%d (hp_iter=%d/%d), set large_found=1\n",
// 						       j, sum, depth, num_client_cur_iters[hp_client], num_client_max_iters[hp_client]);
// 					}

// 					// Clear dedup cache because this kernel is moving forward
// 					clear_skip_cache_for_client(j);

// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					status = 0;
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);

// 					streams[j] = 0;
// 					start = j;
// 				}
// 				else {
// 					// Print a concrete reason only when the client had a ready record but was not scheduled
// 					if (schedule_reason == nullptr) {
// 						bool cond_single_or_hp0_or_mem =
// 							(num_clients == 1) || (seen[hp_client] == 0) || is_mem_like;

// 						bool cond_early_or_hp_done =
// 							(num_client_cur_iters[j] <= 10) ||
// 							(num_client_cur_iters[hp_client] >= num_client_max_iters[hp_client]);

// 						bool cond_update_phase = false;
// 						bool cond_update_event_ready = false;
// 						if (seen[hp_client] >= update_start) {
// 							cond_update_event_ready =
// 								(cudaEventQuery(*(events[hp_client][update_start - 1])) == cudaSuccess);
// 							cond_update_phase = (op_info_0.sm_used <= sm_threshold) && cond_update_event_ready;
// 						}

// 						bool cond_sm_fit = (size + op_info_0.sm_used <= sm_threshold);
// 						bool cond_profile_ok =
// 							(op_info_0.profile == -1 || profiles[hp_client] == -1 ||
// 							 (profiles[hp_client] != op_info_0.profile));

// 						bool cond_sm_profile =
// 							(seen[hp_client] > 0) &&
// 							cond_sm_fit &&
// 							cond_profile_ok;

// 						char reason_buf[512];
// 						snprintf(reason_buf, sizeof(reason_buf),
// 						         "none matched | c1=%d c2=%d c3=%d c4=%d | "
// 						         "hp_seen=%d is_mem=%d be_iter=%d hp_iter=%d hp_max=%d "
// 						         "seen_hp>=update_start=%d update_ev_ready=%d sm_fit(%ld<=%d)=%d profile_ok=%d",
// 						         (int)cond_single_or_hp0_or_mem,
// 						         (int)cond_early_or_hp_done,
// 						         (int)cond_update_phase,
// 						         (int)cond_sm_profile,
// 						         seen[hp_client], (int)is_mem_like,
// 						         num_client_cur_iters[j], num_client_cur_iters[hp_client], num_client_max_iters[hp_client],
// 						         (int)(seen[hp_client] >= update_start),
// 						         (int)cond_update_event_ready,
// 						         (long)(size + op_info_0.sm_used), sm_threshold,
// 						         (int)cond_sm_fit,
// 						         (int)cond_profile_ok);

// 						print_nosched_msg_once(j, reason_buf, op_info_0, frecords[j]);
// 					}
// 					// If schedule_reason != nullptr and schedule became false later, it was already logged by large_found gate path
// 				}
// 			}
// 			// no print when frecords[j] == NULL (not ready record)
// 		}

// 		int finished = 0;
// 		for (int i = 0; i < num_clients; i++) {
// 			//printf("Client %d, seen is %d, all is %d\n", i, seen[i], num_client_kernels[i]);
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i]) ||
// 				(warmup && (num_client_cur_iters[i] == warmup_iters)) ||
// 				(stop_ack[i] == true)
// 			) {
// 				finished += 1;
// 			}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						cudaError_t status = cudaEventQuery(*(events[0][event_ids[0] - 1]));
// 						if (status != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						cudaError_t status = cudaEventQuery(*(events[i][event_ids[i] - 1]));
// 						if (status != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					DEBUG_PRINT("UNLOCK CLIENT %d\n", i);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;

// 					// Clear dedup cache when iteration/kernel state resets
// 					clear_skip_cache_for_client(i);

// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 					if (!seq && i == hp_client && is_train[hp_client]) {
// 						printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration);
// 						hp_iter_duration += duration;
// 						if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
// 							float hp_avg_duration = hp_iter_duration / 10.0;
// 							printf("--------------------- Average iter duration for client 1 is %f ms, limit is %f ms, sm_threshold is %d\n",
// 							       hp_avg_duration, hp_limit_float, sm_threshold);
// 							hp_iter_duration = 0;

// 							// TODO: add better stopping conditions
// 							if (hp_avg_duration > hp_limit_float) {
// 								high_sms = sm_threshold;
// 								sm_threshold = (low_sms + high_sms) / 2;
// 								printf("[ORION][BS] hp_avg=%f > limit=%f -> high_sms=%d, new sm_threshold=%d\n",
// 								       hp_avg_duration, hp_limit_float, high_sms, sm_threshold);
// 							}
// 							else {
// 								low_sms = sm_threshold;
// 								sm_threshold = (low_sms + high_sms) / 2;
// 								printf("[ORION][BS] hp_avg=%f <= limit=%f -> low_sms=%d, new sm_threshold=%d\n",
// 								       hp_avg_duration, hp_limit_float, low_sms, sm_threshold);
// 							}
// 						}
// 					}
// 					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
// 				}
// 				if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i]) ||
// 					(warmup && (num_client_cur_iters[i] == warmup_iters)) ||
// 					(stop_ack[i] == true)
// 				) {
// 					finished += 1;
// 					if (!warmup) {
// 						auto end_total = std::chrono::high_resolution_clock::now();
// 						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
// 						duration /= 1000.0;
// 						printf("Client %d, Total loop took %f sec\n", i, duration);
// 						if (i == num_clients - 1) {
// 							for (int k = 0; k < num_clients - 1; k++) {
// 								printf("======= Client %d has done %d iterations\n", k, num_client_cur_iters[k]);
// 								if (!locked[k])
// 									pthread_mutex_lock(client_mutexes[k]);
// 								stops[k] = true;
// 								if (!locked[k])
// 									pthread_mutex_unlock(client_mutexes[k]);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}

// 		if (finished == num_clients) {
// 			printf("EXIT LOOP!\n");
// 			break;
// 		}
// 	}

// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }


// Multiple Streams
void* Scheduler::busy_wait_ms(int num_clients, int iter, bool warmup, int warmup_iters,
                                    bool seq, int depth, int hp_limit, int update_start)
{


    printf("Entered busy_wait_profile (MS, no profiling)! Num clients is %d\n", num_clients);
    static int rr_start = 0;
    auto start_total = std::chrono::high_resolution_clock::now();

    // Per-client bookkeeping
    std::vector<bool> total_client_set(num_clients, false);

    while (1) {
        std::vector<func_record*> frecords(num_clients, NULL);

        // ----------------------------
        // Pull one record per client
        // ----------------------------
        for (int i = 0; i < num_clients; i++) {


            // All kernels for this iteration already launched
            // if (seen[i] == num_client_kernels[i])
            //     continue;

            pthread_mutex_lock(client_mutexes[i]);
            volatile int sz = client_buffers[i]->size();
            if (sz > 0) {
                frecords[i] = &(client_buffers[i]->front());
                int cur_iter = num_client_cur_iters[i];

                // First kernel of this iteration: set timestamps
                if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
                    client_starts[i] = std::chrono::high_resolution_clock::now();
                    client_starts_set[i][cur_iter] = true;
                    if (!total_client_set[i]) {
                        total_client_starts[i] = std::chrono::high_resolution_clock::now();
                        total_client_set[i] = true;
                    }
                }
            }
            pthread_mutex_unlock(client_mutexes[i]);
        }

        // ----------------------------
        // Check finished events, release masks
        // ----------------------------
        // bool canSchedule[num_clients];
        // for (int i = 0; i < num_clients; ++i) {
        //     canSchedule[i] = true;
        //     if (event_ids[i] >= 1) {
        //         if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
        //             // Previous kernel finished: release TPC mask
        //             unsetmask(i);
        //         } else {
        //             canSchedule[i] = false;
        //         }
        //     }
        // }


        int start = rr_start;
        int rr_next = (rr_start + 1) % num_clients;   // default: rotate by 1 each pass
        bool any_scheduled = false;

        for (int off = 0; off < num_clients; ++off) {
            int j = (start + off) % num_clients;

            if (frecords[j] == NULL) continue;

            func_record* rec = frecords[j];

            // Compute kernels: only if previous one has finished
            // if (!canSchedule[j]) continue;

            // Non compute records: schedule directly, no mask
            if (rec->type == MALLOC_RECORD ||
                rec->type == MEMCPY_RECORD ||
                rec->type == MEMSET_RECORD ||
                rec->type == FREE_RECORD)
            {
                schedule_kernel(*rec, sched_streams[j], j,
                                events[j][event_ids[j]],
                                seen, event_ids, j);
                pop_from_queue(client_buffers[j], client_mutexes[j], j);

                any_scheduled = true;
                rr_next = (j + 1) % num_clients;   // next pass starts after last scheduled client
                continue;
            }

            // --- Compute record path ---
            // IMPORTANT: use the current kernel index, not [j][j]
            const op_info& op_info_cur = op_info_vector[j][seen[j]];
            // int knee = op_info_cur.knee_tpc;

            // if (num_tpcs < knee) continue;

            // If you really want setmask_O instead of setmask:
            // bool is_long = (op_info_cur.is_short == 0);
            // uint32_t subset = setmask_O(knee, j, is_long);
            // if (is_long && subset == 0) continue;  // strict fail for long kernels

            // If you still want the old behavior, keep setmask(knee, j) here.
            // setmask(knee, j);

            // setmask(HW_NUM_TPCS/num_clients , j);
            setmask(12, j);
            schedule_kernel(*rec, sched_streams[j], j,
                            events[j][event_ids[j]],
                            seen, event_ids, j);
            pop_from_queue(client_buffers[j], client_mutexes[j], j);

            any_scheduled = true;
            rr_next = (j + 1) % num_clients;
        }

        // Update rr start for next scheduling pass
        if (any_scheduled) rr_start = rr_next;
        else rr_start = (rr_start + 1) % num_clients;  // still rotate to avoid sticking

        // ----------------------------
        // Handle iteration completion
        // ----------------------------
        int finished = 0;
        for (int i = 0; i < num_clients; i++) {

            if (   (num_client_cur_iters[i] == num_client_max_iters[i])
                || (warmup && (num_client_cur_iters[i] == warmup_iters))
                || (stop_ack[i] == true))
            {
                finished += 1;
            }
            else if (seen[i] >= num_client_kernels[i]) {

                // Do NOT keep the mutex locked across loop iterations
                pthread_mutex_lock(client_mutexes[i]);
                bool empty = client_buffers[i]->empty();
                pthread_mutex_unlock(client_mutexes[i]);

                // If extra records exist (often cuDNN extra memcpy/memset), keep scheduling them
                if (!empty) {
                    continue;
                }

                bool ready = true;
                if (seq) {
                    if (event_ids[0] >= 1 &&
                        cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess)
                        ready = false;
                } else {
                    if (event_ids[i] >= 1 &&
                        cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess)
                        ready = false;
                }

                if (ready) {
                    unsetmask(i);

                    seen[i] = 0;
                    if (seq) event_ids[0] = 0;
                    event_ids[i] = 0;
                    streams[i] = -1;
                    fidx[i] = 0;

                    request_status[i][num_client_cur_iters[i]] = true;
                    num_client_cur_iters[i] += 1;
                    client_progress[i] = 0;

                    auto end = std::chrono::high_resolution_clock::now();
                    float duration_ms =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count() / 1000.0f;
                    client_durations[i].push_back(duration_ms);
                }
            }

        }

        if (finished == num_clients)
            break;
    }

    if (!warmup) {
        auto end_total = std::chrono::high_resolution_clock::now();
        long long duration_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
        printf("Total loop took %lld ns\n", duration_ns);
    }

    return NULL;
}


//orion reef

void* Scheduler::busy_wait_reef(int num_clients,
                                int iter,
                                bool warmup,
                                int warmup_iters,
                                bool seq,
                                int depth,
                                int hp_limit,
                                int update_start)
{
    printf("Entered busy_wait_profile (REEF, no window)! Num clients is %d\n", num_clients);

    int start0 = 0;
    int start1 = 0;
    
    int prev_large = -1;
    int hp_running = -1;

    bool inf_finished = false;
    bool started      = false;

    std::chrono::time_point<std::chrono::system_clock> start_time;
    auto start_total = std::chrono::high_resolution_clock::now();

    std::vector<bool> total_client_set(num_clients, false);
    std::vector<int>  profiles(num_clients, -1);
    std::vector<int>  cur_sms(num_clients, -1);
    int hp_client = num_clients - 1;  // same convention as original REEF

    bool large_found = false;
    long sum  = 0;
    long size = 0;
    int  start = -1;

    int   low_sms      = 0;
    int   high_sms     = max_sms_clients[0]; // 0 is the lp client
    int   sm_threshold = max_sms_clients[0] / 2;
    float hp_iter_duration = 0.0f;
    float hp_limit_float   = (float)hp_limit;

    if (!is_train[hp_client]) {
        sm_threshold = max_sms;
        update_start = INT_MAX;
    }

    int penalty = 0;

    // =========================
    // Single-phase REEF scheduler (no sync/window)
    // =========================
    while (1) {
        std::vector<func_record*> frecords(num_clients, NULL);
        size = 0;

        // Pull one record per client
        for (int i = 0; i < num_clients; i++) {

            if (seen[i] == num_client_kernels[i])
                continue;

            pthread_mutex_lock(client_mutexes[i]);
            volatile int sz = client_buffers[i]->size();
            if (sz > 0) {
                frecords[i] = &(client_buffers[i]->front());
                int cur_iter = num_client_cur_iters[i];
                if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
                    client_starts[i] = std::chrono::high_resolution_clock::now();
                    client_starts_set[i][cur_iter] = true;
                    if (!total_client_set[i]) {
                        total_client_starts[i] = std::chrono::high_resolution_clock::now();
                        total_client_set[i] = true;
                    }
                }
            }
            pthread_mutex_unlock(client_mutexes[i]);
        }

        // Event completion & mask unset
        // bool canSchedule[num_clients];
        // for (int i = 0; i < num_clients; ++i) {
        //     canSchedule[i] = true;
        //     if (event_ids[i] >= 1) {
        //         if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
        //             unsetmask(i);
        //         } else {
        //             canSchedule[i] = false;
        //         }
        //     }
        // }

        // ---------- Part 1: schedule mem/admin ops (no log, no prints) ----------
        bool scheduled_mem = false;
        for (int i = 0; i < num_clients; i++) {
            if (frecords[i] != NULL) {
                if (frecords[i]->type == MALLOC_RECORD ||
                    frecords[i]->type == MEMCPY_RECORD ||
                    frecords[i]->type == MEMSET_RECORD ||
                    frecords[i]->type == FREE_RECORD) {

                    schedule_kernel(*(frecords[i]), sched_streams[i], i,
                                    events[i][event_ids[i]], seen, event_ids, i);
                    pop_from_queue(client_buffers[i], client_mutexes[i], i);
                    scheduled_mem = true;
                    break;
                }
            }
        }

        // ---------- Part 2: REEF compute scheduling ----------
        if (!scheduled_mem) {

            hp_client = (hp_client + 1) % num_clients;
            long int endT = 0;

            // Try HP first
            // if (frecords[hp_client] != NULL && canSchedule[hp_client]) {
            if (frecords[hp_client] != NULL) {

                if (frecords[hp_client]->type != MALLOC_RECORD &&
                    frecords[hp_client]->type != MEMCPY_RECORD &&
                    frecords[hp_client]->type != MEMSET_RECORD &&
                    frecords[hp_client]->type != FREE_RECORD) {

                    int hp_idx = seen[hp_client];
                    op_info op_info_1 = op_info_vector[hp_client][hp_idx];

                    int tpc_usage = op_info_1.sm_used / 2;
                    tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;

                    if (num_tpcs >= tpc_usage) {
                        // setmask(tpc_usage, hp_client);

                        schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client],
                                        hp_client, events[hp_client][event_ids[hp_client]],
                                        seen, event_ids, hp_client);
                        pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);

                        int   dur          = op_info_1.duration;
                        float threshold    = 0.8f;
                        auto  current_time = std::chrono::high_resolution_clock::now();
                        auto  duration_ns  =
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                current_time.time_since_epoch()).count();
                        endT = static_cast<long int>(
                            static_cast<double>(dur) * threshold + static_cast<double>(duration_ns));
                    }
                } else {
                    // HP mem/admin (handled above normally)
                    schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client],
                                    hp_client, events[hp_client][event_ids[hp_client]],
                                    seen, event_ids, hp_client);
                    pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
                }

                // If HP scheduled, schedule BE kernels
                if (endT > 0) {

                    for (int i = 0; i < num_clients; i++) {
                        if (i == hp_client) continue;
                        if (frecords[i] == NULL) continue;

                        if (frecords[i]->type == MALLOC_RECORD ||
                            frecords[i]->type == MEMCPY_RECORD ||
                            frecords[i]->type == MEMSET_RECORD ||
                            frecords[i]->type == FREE_RECORD) {
                            continue;
                        }

                        // if (!canSchedule[i]) continue;

                        op_info op_info_0 = op_info_vector[i][seen[i]];
                        int tpc_usage = op_info_0.sm_used / 2;
                        tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;

                        if (num_tpcs > 0) {
                            int assigned = std::min(num_tpcs, tpc_usage);
                            // setmask(assigned, i);

                            schedule_kernel(*(frecords[i]), sched_streams[i], i,
                                            events[i][event_ids[i]],
                                            seen, event_ids, i);
                            pop_from_queue(client_buffers[i], client_mutexes[i], i);
                        }
                    }
                }

            } else {

                // HP not scheduled: penalty handling
                bool any_BE = false;
                for (int i = 0; i < num_clients; i++) {
                    if (frecords[i] != NULL && i != hp_client &&
                        frecords[i]->type != MALLOC_RECORD &&
                        frecords[i]->type != MEMCPY_RECORD &&
                        frecords[i]->type != MEMSET_RECORD &&
                        frecords[i]->type != FREE_RECORD) {
                        any_BE = true;
                    }
                }
                if (any_BE) penalty += 1;

                if (penalty >= depth) {
                    // schedule all BE
                    for (int i = 0; i < num_clients; i++) {
                        // if (frecords[i] != NULL && canSchedule[i] && i != hp_client) {
                        if (frecords[i] != NULL && i != hp_client) {
                            if (frecords[i]->type == MALLOC_RECORD ||
                                frecords[i]->type == MEMCPY_RECORD ||
                                frecords[i]->type == MEMSET_RECORD ||
                                frecords[i]->type == FREE_RECORD) {
                                continue;
                            }

                            op_info op_info_0 = op_info_vector[i][seen[i]];
                            int tpc_usage = op_info_0.sm_used / 2;
                            tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;

                            if (num_tpcs > 0) {
                                int assigned = std::min(num_tpcs, tpc_usage);
                                // setmask(assigned, i);

                                schedule_kernel(*(frecords[i]), sched_streams[i], i,
                                                events[i][event_ids[i]],
                                                seen, event_ids, i);
                                pop_from_queue(client_buffers[i], client_mutexes[i], i);
                            }
                        }
                    }
                    penalty = 0;
                }
            }
        } // end REEF compute scheduling

        // ---------- Finished check ----------
        int finished = 0;
        for (int i = 0; i < num_clients; i++) {

            if ((num_client_cur_iters[i] == num_client_max_iters[i])
                || (warmup && (num_client_cur_iters[i] == warmup_iters))
                || (stop_ack[i] == true)) {
                finished += 1;
            }
            else if (seen[i] == num_client_kernels[i]) {
                if (!locked[i]) {
                    pthread_mutex_lock(client_mutexes[i]);
                    locked[i] = true;
                }
                bool ready = true;
                if (seq) {
                    if (event_ids[0] >= 1) {
                        if (cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess)
                            ready &= false;
                    }
                }
                else {
                    if (event_ids[i] >= 1) {
                        if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess)
                            ready &= false;
                    }
                }
                if (ready) {
                    // unsetmask(i);
                    seen[i] = 0;
                    if (seq)
                        event_ids[0] = 0;
                    event_ids[i] = 0;
                    streams[i]   = -1;
                    fidx[i]      = 0;
                    request_status[i][num_client_cur_iters[i]] = true;
                    pthread_mutex_unlock(client_mutexes[i]);
                    num_client_cur_iters[i] += 1;
                    locked[i] = false;

                    auto end = std::chrono::high_resolution_clock::now();
                    float duration =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            end - client_starts[i]).count() / 1000.0f;
                    client_durations[i].push_back(duration);
                }
                if ((num_client_cur_iters[i] == num_client_max_iters[i])
                    || (warmup && (num_client_cur_iters[i] == warmup_iters))
                    || (stop_ack[i] == true)) {
                    finished += 1;
                }
            }
        }

        if (finished == num_clients) {
            printf("[REEF] All clients finished, exit\n");
            break;
        }
    } // while(1)

    if (!warmup) {
        auto end_total = std::chrono::high_resolution_clock::now();
        float duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end_total - start_total).count();
        duration /= 1000.0f;
        printf("Total loop took %f sec\n", duration);
    }

    return NULL;
}

// void* Scheduler::busy_wait_reef(int num_clients,
//                                 int iter,
//                                 bool warmup,
//                                 int warmup_iters,
//                                 bool seq,
//                                 int depth,
//                                 int hp_limit,
//                                 int update_start)
// {

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	int hp_client = num_clients-1;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

// 	// if hp is inference, use max_sms + also there is no update phase
// 	if (!is_train[hp_client]) {
// 		sm_threshold = max_sms;
// 		update_start = INT_MAX;
// 	}

// 	while(1) {
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {
// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 				//if (seen[i] == num_client_kernels[i]-1)
// 				//	continue;
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}

//         hp_client = (hp_client + 1) % num_clients;
//         schedule_reef(frecords, num_clients, depth, hp_client);
		
// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				finished += 1;
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
//                     unsetmask(i);
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;

// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
					
// 					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
// 				}
// 				if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				) {
// 					finished += 1;
// 					if (!warmup) {
// 						auto end_total = std::chrono::high_resolution_clock::now();
// 						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
// 						duration /= 1000.0;
// 						printf("Client %d, Total loop took %f sec\n", i, duration);
// 					}
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;

// 	}
// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }




// KRISP
void* Scheduler::busy_wait_krisp(int num_clients,
                                 int iter,
                                 bool warmup,
                                 int warmup_iters,
                                 bool seq,
                                 int depth,
                                 int hp_limit,
                                 int update_start)
{
    (void)iter;
    (void)warmup_iters;
    (void)depth;
    (void)hp_limit;
    (void)update_start;

    DEBUG_PRINT("Entered busy_wait_krisp! Num clients = %d\n", num_clients);

    auto start_total = std::chrono::high_resolution_clock::now();
    std::vector<bool> total_client_set(num_clients, false);

    // ---- knee-TPC picker using op_info.knee_tpc ----
    auto pick_knee_tpc = [&](const op_info& op) -> int {
        int knee = op.knee_tpc;      // make sure op_info has this field
        if (knee < 1) knee = 1;
        if (knee > num_tpcs) knee = num_tpcs;
        return knee;
    };

    while (true) {


        std::vector<func_record*> frecords(num_clients, nullptr);

        // 1) pull record for each client
        for (int i = 0; i < num_clients; ++i) {
            if (is_executing[i]) continue;
            if (seen[i] == num_client_kernels[i]) continue;

            pthread_mutex_lock(client_mutexes[i]);
            volatile int sz = client_buffers[i]->size();
            if (sz > 0) {
                frecords[i] = &client_buffers[i]->front();
                int cur_iter = num_client_cur_iters[i];
                if (seen[i] == 0 && !client_starts_set[i][cur_iter]) {
                    client_starts[i] = std::chrono::high_resolution_clock::now();
                    client_starts_set[i][cur_iter] = true;
                    if (!total_client_set[i]) {
                        total_client_starts[i] = std::chrono::high_resolution_clock::now();
                        total_client_set[i] = true;
                    }
                }
            }
            pthread_mutex_unlock(client_mutexes[i]);
        }

        // 2) check events & build canSchedule[]
        std::vector<bool> canSchedule(num_clients, true);
        for (int i = 0; i < num_clients; ++i) {
            if (event_ids[i] >= 1) {
                if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                    unsetmask(i);
                } else {
                    canSchedule[i] = false;
                }
            }
        }

        // 3) KRISP scheduling
        // ---------------------------
        // 3.1 Admin ops first
        auto is_admin_record = [&](const func_record& r) {
            return (r.type == MALLOC_RECORD ||
                    r.type == MEMCPY_RECORD ||
                    r.type == MEMSET_RECORD ||
                    r.type == FREE_RECORD);
        };

 
        for (int j = 0; j < num_clients; ++j) {
            if (frecords[j] != NULL) {

                if (frecords[j]->type != MALLOC_RECORD &&
                    frecords[j]->type != MEMCPY_RECORD &&
                    frecords[j]->type != MEMSET_RECORD &&
                    frecords[j]->type != FREE_RECORD) {

                    if (canSchedule[j] == false) {
                        continue;
                    }

                    // Compute kernel:
                    // 1) Decide TPCs to use via knee-tpc hint
                    op_info opj        = op_info_vector[j][seen[j]];
                    int     knee_tpc   = opj.knee_tpc;
                    int     assigned_tpc = min(num_tpcs, knee_tpc);
                    // printf("knee is %d\n , assign is %d", opj.knee_tpc, assigned_tpc);
                    if (assigned_tpc < 1) continue; // safety
                    
                    // 3) Set the mask per your hint
                    setmask_krisp(assigned_tpc, j);

                    // 4) Schedule
                    schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                    events[j][event_ids[j]], seen, event_ids, j);
                    pop_from_queue(client_buffers[j], client_mutexes[j], j);
                } else {
                    // mem/admin kernel: schedule without logging and without mask
                    schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                    events[j][event_ids[j]], seen, event_ids, j);
                    pop_from_queue(client_buffers[j], client_mutexes[j], j);
                }
            }
        }

        // 4) iteration completion
        int finished = 0;
        for (int i = 0; i < num_clients; ++i) {

            if (   num_client_cur_iters[i] == num_client_max_iters[i]
                || (warmup && num_client_cur_iters[i] == warmup_iters)
                || stop_ack[i])
            {
                finished += 1;
                continue;
            }

            if (seen[i] == num_client_kernels[i]) {
                if (!locked[i]) {
                    pthread_mutex_lock(client_mutexes[i]);
                    locked[i] = true;
                    DEBUG_PRINT("LOCK CLIENT %d\n", i);
                }

                bool ready = true;
                if (seq) {
                    if (event_ids[0] >= 1 &&
                        cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess)
                    {
                        ready = false;
                    }
                } else {
                    if (event_ids[i] >= 1 &&
                        cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess)
                    {
                        ready = false;
                    }
                }

                if (ready) {
                    unsetmask(i);

                    seen[i] = 0;
                    if (seq) event_ids[0] = 0;
                    event_ids[i] = 0;
                    streams[i] = -1;
                    fidx[i]    = 0;
                    request_status[i][num_client_cur_iters[i]] = true;
                    pthread_mutex_unlock(client_mutexes[i]);
                    num_client_cur_iters[i] += 1;
                    locked[i] = false;
                    client_progress[i] = 0;

                    auto end = std::chrono::high_resolution_clock::now();
                    float duration_ms =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            end - client_starts[i]).count() / 1000.0f;
                    client_durations[i].push_back(duration_ms);
                }
            }
        }

        if (finished == num_clients) break;
    }

    if (!warmup) {
        auto end_total = std::chrono::high_resolution_clock::now();
        long long duration_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_total - start_total).count();
        printf("Total KRISP loop took %lld ns\n", duration_ns);
    }

    return NULL;
}


// in test ms
// void* Scheduler::busy_wait_ms(int num_clients, int iter, bool warmup, int warmup_iters,
//                              bool seq, int depth, int hp_limit, int update_start)
// {
//     DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);

//     using Clock = std::chrono::high_resolution_clock;
//     auto start_total = Clock::now();

//     while (1) {

//         std::vector<func_record*> frecords(num_clients, NULL);

//         for (int i = 0; i < num_clients; i++) {

//             if (is_executing[i] == true) continue;
//             if (seen[i] == num_client_kernels[i]) continue;

//             pthread_mutex_lock(client_mutexes[i]);
//             volatile int sz = client_buffers[i]->size();
//             if (sz > 0) {
//                 frecords[i] = &(client_buffers[i]->front());
//                 int cur_iter = num_client_cur_iters[i];
//                 if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
//                     client_starts[i] = std::chrono::high_resolution_clock::now();
//                     client_starts_set[i][cur_iter] = true;
//                 }
//             }
//             pthread_mutex_unlock(client_mutexes[i]);
//         }

//         // Build list of compute-ready clients whose LAST GPU kernel finished
//         std::vector<int> ready_client;
//         ready_client.reserve(num_clients);

//         bool canSchedule[num_clients];
//         for (int i = 0; i < num_clients; ++i) {
//             canSchedule[i] = true;
//             if (event_ids[i] >= 1) {
//                 if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
//                     // Previous kernel finished: release TPC mask
//                     unsetmask(i);
//                 } else {
//                     canSchedule[i] = false;
//                 }
//             }
//         }

//         for (int j = 0; j < num_clients; ++j) {

//             if (frecords[j] == NULL) continue;
//             if (!canSchedule) continue;
            
//             // non-compute ops: schedule immediately
//             if (frecords[j]->type == MALLOC_RECORD ||
//                 frecords[j]->type == MEMCPY_RECORD ||
//                 frecords[j]->type == MEMSET_RECORD ||
//                 frecords[j]->type == FREE_RECORD ||
//                 warmup)
//             {
//                 schedule_kernel(*(frecords[j]), sched_streams[j], j,
//                                 events[j][event_ids[j]], seen, event_ids, j);
//                 pop_from_queue(client_buffers[j], client_mutexes[j], j);
//                 continue;
//             }
//             else{
//                 ready_client.push_back(j);
//             }
//         }

//         // ------------------------------------------------------------
//         // BATCH COMPUTE SCHEDULING POLICY:
//         //   (A) normal: only schedule compute when >= 2 clients are ready
//         //   (B) drain: if only 1 client is ready, schedule it only if
//         //       all other clients are finished (no more iterations / stop_ack / warmup done)
//         // ------------------------------------------------------------
//         if (!warmup) {

//             auto client_done = [&](int i) -> bool {
//                 if (num_client_cur_iters[i] == num_client_max_iters[i]) return true;
//                 if (warmup && (num_client_cur_iters[i] == warmup_iters)) return true;
//                 if (stop_ack[i] == true) return true;
//                 return false;
//             };

//             if ((int)ready_client.size() > 1) {
//                 // Normal batch: schedule all ready compute clients together
//                 for (int client_id : ready_client) {
//                     schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id,
//                                     events[client_id][event_ids[client_id]],
//                                     seen, event_ids, client_id);
//                     pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
//                 }
//             }
//             else if ((int)ready_client.size() == 1) {
//                 // Drain exception: only schedule if no other client can still do work overall
//                 int cid = ready_client[0];

//                 bool all_others_done = true;
//                 for (int k = 0; k < num_clients; ++k) {
//                     if (k == cid) continue;
//                     // if (!client_done(k) || seen[k] < num_client_kernels[k]) {
//                     //     all_others_done = false;
//                     //     break;
//                     // }
//                     if (seen[k] < num_client_kernels[k]) {
//                         all_others_done = false;
//                         break;
//                     }
//                 }

//                 if (all_others_done) {
//                     schedule_kernel(*(frecords[cid]), sched_streams[cid], cid,
//                                     events[cid][event_ids[cid]],
//                                     seen, event_ids, cid);
//                     pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);
//                 }
//             }
//         }

//         int finished = 0;
//         for (int i = 0; i < num_clients; i++) {

//             if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
//                 (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
//                 (stop_ack[i] == true))
//             {
//                 finished += 1;
//             }
//             else if (seen[i] == num_client_kernels[i]) {

//                 if (!locked[i]) {
//                     pthread_mutex_lock(client_mutexes[i]);
//                     locked[i] = true;
//                     DEBUG_PRINT("LOCK CLIENT %d\n", i);
//                 }

//                 bool ready = true;
//                 if (seq) {
//                     if (event_ids[0] >= 1) {
//                         if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
//                             ready &= false;
//                     }
//                 } else {
//                     if (event_ids[i] >= 1) {
//                         if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
//                             ready &= false;
//                     }
//                 }

//                 if (ready) {
//                     unsetmask(i);

//                     seen[i] = 0;
//                     if (seq) event_ids[0] = 0;
//                     event_ids[i] = 0;
//                     streams[i] = -1;
//                     fidx[i] = 0;
//                     request_status[i][num_client_cur_iters[i]] = true;

//                     pthread_mutex_unlock(client_mutexes[i]);
//                     num_client_cur_iters[i] += 1;
//                     locked[i] = false;
//                     client_progress[i] = 0;

//                     auto end = std::chrono::high_resolution_clock::now();
//                     float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
//                     duration /= 1000.0; // ms
//                     client_durations[i].push_back(duration);
//                 }
//             }
//         }

//         if (finished == num_clients) break;
//     }

//     if (!warmup) {
//         auto end_total = std::chrono::high_resolution_clock::now();
//         double duration_ms =
//             std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count() / 1000.0;
//         printf("Total loop took %.3f ms\n", duration_ms);
//     }

//     return NULL;
// }






// ------------------------------------------------------------------
// Existing
// ------------------------------------------------------------------
static inline float excl_at_n(const op_info* op, int n) {
    if (!op) return NAN;
    if (n >= 0 && n < (int)op->excl_us_by_n.size()) {
        float v = op->excl_us_by_n[n];
        return (std::isfinite(v) && v > 0.f) ? v : NAN;
    }
    return NAN;
}

static inline uint64_t now_ns_hrc() {
    using Clock = std::chrono::high_resolution_clock;
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}

// no-mask => full device
static inline int normalize_tpc_used(int tpc_raw) {
    if (tpc_raw <= 0 || tpc_raw > HW_NUM_TPCS) return HW_NUM_TPCS;
    return tpc_raw;
}

static inline float estimate_excl_us(const op_info* op, int tpc_used_norm) {
    float us = excl_at_n(op, tpc_used_norm);

    if (!std::isfinite(us) && op && op->opt_tpc_exclusive > 0) {
        us = excl_at_n(op, op->opt_tpc_exclusive);
    }
    if (!std::isfinite(us) && op && op->duration > 0.f) {
        // duration is ms in your struct -> convert to us
        us = op->duration * 1000.f;
    }
    if (!std::isfinite(us) || us <= 0.f) us = 1000.f; // final fallback 1ms
    return us;
}

static inline int popcount_mask(uint32_t x) {
    constexpr uint32_t USED_BITS = (HW_NUM_TPCS == 32)
        ? 0xFFFFFFFFu
        : ((1u << HW_NUM_TPCS) - 1u);
    return __builtin_popcount(x & USED_BITS);
}

static void shrink_mask_for_client(int idx, int needed_tpcs)
{
    using tpc_mask_t = uint32_t;
    tpc_mask_t owned = localMask[idx];   // use your real mask array name
    int have = popcount_mask(owned);

    if (have <= needed_tpcs) return;

    int to_release = have - needed_tpcs;

    for (int i = HW_NUM_TPCS - 1; i >= 0 && to_release > 0; --i) {
        if ((owned >> i) & tpc_mask_t(1)) {
            owned &= ~(tpc_mask_t(1) << i);

            // keep consistent with your mask semantics
            mask &= ~(tpc_mask_t(1) << i);

            if (tpc_usage_count[i] > 0) --tpc_usage_count[i];
            ++num_tpcs; // if this means free TPCs in your codebase
            --to_release;
        }
    }

    localMask[idx] = owned;
}

static inline int client_active_tpcs(int client_id)
{
    int tpcs = 0;
    for (const auto& rk : g_running_ops) {
        if (rk.client_id == client_id && rk.op && rk.op->is_running) {
            tpcs = std::max(tpcs, normalize_tpc_used(rk.op->tpc_used));
        }
    }
    return tpcs;
}

// push/update any scheduled kernel (long or short)
static inline void mark_kernel_running(
    int client_id,
    op_info* op,
    int tpc_used_raw,   // pass -1 for no-mask mode
    uint64_t now_ns)
{
    if (!op) return;

    const int tpc_used = normalize_tpc_used(tpc_used_raw); // no-mask => full
    const float excl_us = estimate_excl_us(op, tpc_used);
    const uint64_t est_dur_ns = (uint64_t)std::llround((double)excl_us * 1.2 * 1000.0); // us->ns

    op->tpc_used      = tpc_used;
    op->is_running    = true;
    op->est_start_ns  = now_ns;
    op->est_finish_ns = now_ns + est_dur_ns;

    auto it = std::find_if(g_running_ops.begin(), g_running_ops.end(),
        [&](const RunningOp& rk) {
            return rk.client_id == client_id && rk.kernel_id == op->id;
        });

    if (it == g_running_ops.end()) {
        g_running_ops.push_back(RunningOp{client_id, op->id, op});
    } else {
        it->op = op; // refresh pointer
    }
}

// reap finished ops, build currently-running list, and unset/shrink masks once per client
static inline void update_and_build_running_ops(
    uint64_t now_ns,
    int exclude_client_id,
    int num_clients,
    std::vector<const op_info*>& out_running)
{
    auto& vec = g_running_ops;

    out_running.clear();
    out_running.reserve(vec.size());

    std::vector<RunningOp> new_vec;
    new_vec.reserve(vec.size());

    std::vector<int> needed_tpcs(num_clients, 0);

    auto finished = [&](const RunningOp& rk) -> bool {
        op_info* op = rk.op;
        if (!op) return true;
        if (!op->is_running) return true;
        if (op->est_finish_ns > 0 && op->est_finish_ns <= now_ns) return true;
        return false;
    };

    for (auto& rk : vec) {
        if (finished(rk)) {
            if (rk.op) {
                rk.op->is_running = false;
                rk.op->est_start_ns = 0;
                rk.op->est_finish_ns = 0;
            }
            continue;
        }

        new_vec.push_back(rk);

        if (rk.client_id >= 0 && rk.client_id < num_clients && rk.op) {
            needed_tpcs[rk.client_id] =
                std::max(needed_tpcs[rk.client_id], normalize_tpc_used(rk.op->tpc_used));
        }

        if (rk.client_id != exclude_client_id && rk.op) {
            out_running.push_back(rk.op);
        }
    }

    // apply mask maintenance once per client
    for (int cid = 0; cid < num_clients; ++cid) {
        const int need = needed_tpcs[cid];
        if (need <= 0) {
            unsetmask(cid);
        } else {
            shrink_mask_for_client(cid, need);
        }
    }

    vec.swap(new_vec);
}

static inline void clear_running_ops_for_client(int client_id) {
    for (auto& rk : g_running_ops) {
        if (rk.client_id == client_id && rk.op) {
            rk.op->is_running = false;
            rk.op->est_start_ns = 0;
            rk.op->est_finish_ns = 0;
        }
    }
    g_running_ops.erase(
        std::remove_if(g_running_ops.begin(), g_running_ops.end(),
            [&](const RunningOp& rk){ return rk.client_id == client_id; }),
        g_running_ops.end());
}



std::vector<std::pair<Scheduler::CriticalKernel,Scheduler::CriticalKernel>>
load_profile_plan(const std::string& csv_path)
{
    std::vector<std::pair<Scheduler::CriticalKernel,Scheduler::CriticalKernel>> plan;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << csv_path << "\n";
        return plan;
    }

    std::string line;
    // skip header
    if (!std::getline(file, line)) return plan;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        int id1, id2, tpc1, tpc2;
        char comma;
        // parse "id1,id2,tpc1,tpc2,duration"
        if (ss >> id1 >> comma
               >> id2 >> comma
               >> tpc1 >> comma
               >> tpc2)
        {
            Scheduler::CriticalKernel k1{ id1, tpc1 };
            Scheduler::CriticalKernel k2{ id2, tpc2 };
            plan.emplace_back(k1, k2);
        }
    }
    return plan;
}


static inline void push_running_op_unique(int client_id, int kernel_id, op_info* op)
{
    auto& rops = g_running_ops;

    // remove any duplicated (client,kernel) OR duplicated pointer
    for (auto it = rops.begin(); it != rops.end();) {
        const bool same_pair = (it->client_id == client_id && it->kernel_id == kernel_id);
        const bool same_ptr  = (it->op == op);
        if (same_pair || same_ptr) {
            it = rops.erase(it);
        } else {
            ++it;
        }
    }

    rops.push_back({client_id, kernel_id, op});
}

static inline void clear_client_running_ops(int client_id)
{
    auto& rops = g_running_ops;
    for (auto it = rops.begin(); it != rops.end();) {
        if (it->client_id == client_id) {
            if (it->op) it->op->is_running = false;
            it = rops.erase(it);
        } else {
            ++it;
        }
    }
}




// ============================================================
// Split multiplier LUT: (clusterA, clusterB, tpcA, tpcB) -> mult
// CSV format:
// cluster_ut,cluster_co,split,tpc1,tpc2,mult_geomean,n_samples,std_log
// ============================================================

struct SplitKey {
    int cu, cc, t1, t2;
    bool operator==(const SplitKey& o) const {
        return cu == o.cu && cc == o.cc && t1 == o.t1 && t2 == o.t2;
    }
};
struct SplitKeyHash {
    size_t operator()(const SplitKey& k) const noexcept {
        size_t h = 1469598103934665603ULL;
        auto mix = [&](uint64_t x) {
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix((uint32_t)k.cu);
        mix((uint32_t)k.cc);
        mix((uint32_t)(int32_t)k.t1);
        mix((uint32_t)(int32_t)k.t2);
        return h;
    }
};

static std::unordered_map<SplitKey, float, SplitKeyHash> g_split_mult;
static bool g_split_mult_loaded = false;

static inline std::string trim_copy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) b++;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
}

static inline std::vector<std::string> split_csv_line_quote_ok(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quote = false;
    for (char c : line) {
        if (c == '"') { in_quote = !in_quote; continue; }
        if (c == ',' && !in_quote) {
            out.push_back(trim_copy(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(trim_copy(cur));
    return out;
}

static void load_split_mult_once(const char* csv_path) {
    if (g_split_mult_loaded) return;
    g_split_mult_loaded = true;

    std::ifstream fin(csv_path);
    if (!fin.is_open()) {
        std::fprintf(stderr, "[WARN] cannot open split-mult csv: %s\n", csv_path);
        return;
    }

    std::string header;
    if (!std::getline(fin, header)) return;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto cols = split_csv_line_quote_ok(line);
        if (cols.size() < 6) continue;

        int cu, cc, t1, t2;
        float mult;
        try {
            cu   = std::stoi(cols[0]);
            cc   = std::stoi(cols[1]);
            t1   = std::stoi(cols[3]);
            t2   = std::stoi(cols[4]);
            mult = std::stof(cols[5]);
        } catch (...) {
            continue;
        }

        g_split_mult[{cu, cc, t1, t2}] = mult;
    }
}

static inline bool lookup_split_mult(int cu, int cc, int t1, int t2, float& out_mult) {
    auto it = g_split_mult.find(SplitKey{cu, cc, t1, t2});
    if (it != g_split_mult.end()) { out_mult = it->second; return true; }

    // symmetry fallback
    it = g_split_mult.find(SplitKey{cc, cu, t2, t1});
    if (it != g_split_mult.end()) { out_mult = it->second; return true; }

    return false;
}

// excl_us_by_n is indexed by exact TPC count (you push NaN at index 0).
static inline float excl_us_at_tpc_or_full(const op_info& op, int tpc) {
    int eff = tpc;
    if (eff < 0) eff = HW_NUM_TPCS; // (-1) means no mask => use full TPC count
    if (eff < 1) eff = 1;
    if (eff > HW_NUM_TPCS) eff = HW_NUM_TPCS;

    if (eff >= 0 && eff < (int)op.excl_us_by_n.size()) {
        float v = op.excl_us_by_n[eff];
        if (std::isfinite(v) && v > 0.f) return v;
    }
    return std::numeric_limits<float>::infinity();
}

static inline void build_candidate_splits(std::vector<std::pair<int,int>>& out) {
    out.clear();
    out.reserve(16);
    out.push_back({-1,-1}); // no mask
    out.push_back({1, HW_NUM_TPCS - 1});
    for (int t1 = 2; t1 < HW_NUM_TPCS; t1 += 2) {
        int t2 = HW_NUM_TPCS - t1;
        if (t2 > 0) out.push_back({t1, t2});
    }
}


//ficks
// priority queue 
// prediction model
// void* Scheduler::busy_wait_ficks(int num_clients, int iter, bool warmup, int warmup_iters,
//                                  bool seq, int depth, int hp_limit, int update_start)
// {
//     DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);

//     depth = 10;
//     using Clock = std::chrono::high_resolution_clock;
//     auto start_total = Clock::now();

//     // ============================================================
//     // Decision debug (prints only a bounded number of decisions)
//     // ============================================================
//     const bool DBG_DECISIONS = false;   // flip to false to silence
//     const int  DBG_EVERY     = 1;       // print every N decisions
//     const long DBG_MAX       = 2000;    // hard cap to avoid log storms
//     static long dbg_cnt = 0;

//     auto DBG_OK = [&]() -> bool {
//         if (!DBG_DECISIONS) return false;
//         if (warmup) return false;
//         if (dbg_cnt >= DBG_MAX) return false;
//         return ((dbg_cnt % DBG_EVERY) == 0);
//     };

//     auto print_split = [&](int cidA, int cidB,
//                            const op_info& opA, const op_info& opB,
//                            int tA, int tB, bool ok, float mult,
//                            float exclA, float exclB, float pred,
//                            bool is_best)
//     {
//         if (!DBG_OK()) return;

//         // note: tA/tB == -1 means no mask
//         printf("[FICKS-DEC] cand  cidA=%d(kidx=%d,id=%d,name=%s,cl=%d)  "
//                "cidB=%d(kidx=%d,id=%d,name=%s,cl=%d)  "
//                "split=(%d,%d)  ok=%d  mult=%.6f  exclA=%.3f  exclB=%.3f  "
//                "max=%.3f  pred=%.3f%s\n",
//                cidA, (int)(opA.id), opA.id, opA.name.c_str(), opA.cluster,
//                cidB, (int)(opB.id), opB.id, opB.name.c_str(), opB.cluster,
//                tA, tB, ok ? 1 : 0, (double)mult,
//                (double)exclA, (double)exclB, (double)std::max(exclA, exclB),
//                (double)pred,
//                is_best ? "  <-- best" : "");
//         fflush(stdout);
//     };

//     auto schedule_one = [&](int cid, std::vector<func_record*>& frecords) {
//         schedule_kernel(*(frecords[cid]),
//                         sched_streams[cid],
//                         cid,
//                         events[cid][event_ids[cid]],
//                         seen, event_ids, cid);
//         pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);
//     };

//     auto pick_best_split_for_two = [&](int cidA, int cidB,
//                                    op_info& opA, op_info& opB,
//                                    int& out_tA, int& out_tB,
//                                    float& out_best_pred,
//                                    float& out_best_mult,
//                                    float& out_best_exclA,
//                                    float& out_best_exclB)
//     {
//         const int ca = opA.cluster;
//         const int cb = opB.cluster;

//         std::vector<std::pair<int,int>> splits;
//         build_candidate_splits(splits);

//         float best_pred  = std::numeric_limits<float>::infinity();
//         int   best_tA    = -1;
//         int   best_tB    = -1;
//         float best_mult  = 1.0f;
//         float best_exclA = NAN;
//         float best_exclB = NAN;

//         for (const auto& sp : splits) {
//             int tA = sp.first;
//             int tB = sp.second;

//             float mult = 1.0f;
//             bool ok = lookup_split_mult(ca, cb, tA, tB, mult);

//             // allow (-1,-1) even if LUT missing
//             if (!ok && !(tA == -1 && tB == -1)) continue;

//             float exclA = excl_us_at_tpc_or_full(opA, tA);
//             float exclB = excl_us_at_tpc_or_full(opB, tB);
//             float pred  = mult * std::max(exclA, exclB);

//             bool is_best = false;
//             if (pred < best_pred) {
//                 best_pred  = pred;
//                 best_tA    = tA;
//                 best_tB    = tB;
//                 best_mult  = mult;
//                 best_exclA = exclA;
//                 best_exclB = exclB;
//                 is_best    = true;
//             }

//             print_split(cidA, cidB, opA, opB, tA, tB, ok, mult, exclA, exclB, pred, is_best);
//         }

//         // safe fallback
//         if (!std::isfinite(best_pred)) {
//             best_tA    = -1;
//             best_tB    = -1;
//             best_mult  = 1.0f;
//             best_exclA = excl_us_at_tpc_or_full(opA, -1);
//             best_exclB = excl_us_at_tpc_or_full(opB, -1);
//             best_pred  = best_mult * std::max(best_exclA, best_exclB);
//         }

//         // NEW RULE:
//         // if chosen split is "too close" (difference <= 4), use no-mask for both
//         // (only apply when both are real masked tpc values)
//         constexpr int kSplitDiffToNoMask = 4;
//         if (best_tA > 0 && best_tB > 0 && std::abs(best_tA - best_tB) <= kSplitDiffToNoMask) {
//             float nm_mult = 1.0f;
//             bool nm_ok = lookup_split_mult(ca, cb, -1, -1, nm_mult);
//             if (!nm_ok) nm_mult = 1.0f; // keep current behavior

//             best_tA    = -1;
//             best_tB    = -1;
//             best_mult  = nm_mult;
//             best_exclA = excl_us_at_tpc_or_full(opA, -1);
//             best_exclB = excl_us_at_tpc_or_full(opB, -1);
//             best_pred  = best_mult * std::max(best_exclA, best_exclB);

//             if (DBG_OK()) {
//                 printf("[FICKS-DEC] override: |tA-tB|<=%d => force NO_MASK (-1,-1)\n",
//                     kSplitDiffToNoMask);
//                 fflush(stdout);
//             }
//         }

//         out_tA         = best_tA;
//         out_tB         = best_tB;
//         out_best_pred  = best_pred;
//         out_best_mult  = best_mult;
//         out_best_exclA = best_exclA;
//         out_best_exclB = best_exclB;
//     };

//     // ============================================================
//     // Running-ops bookkeeping (push BOTH long/short; no-mask => all TPC)
//     // ============================================================
//     const int HW_TPCS = 24; // change if needed on non-A4000

//     auto now_ns = [&]() -> uint64_t {
//         return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
//             Clock::now().time_since_epoch()).count();
//     };

//     auto normalize_tpc_used = [&](int tpc_raw) -> int {
//         if (tpc_raw <= 0 || tpc_raw > HW_TPCS) return HW_TPCS; // no-mask mode uses all TPC
//         return tpc_raw;
//     };

//     auto estimate_excl_us = [&](const op_info* op, int tpc_used_norm) -> float {
//         if (!op) return 1000.0f;

//         float us = NAN;
//         if (tpc_used_norm >= 0 && tpc_used_norm < (int)op->excl_us_by_n.size()) {
//             float v = op->excl_us_by_n[tpc_used_norm];
//             if (std::isfinite(v) && v > 0.0f) us = v;
//         }

//         if (!std::isfinite(us)) {
//             us = excl_us_at_tpc_or_full(*op, tpc_used_norm);
//         }

//         if ((!std::isfinite(us) || us <= 0.0f) && op->opt_tpc_exclusive > 0 &&
//             op->opt_tpc_exclusive < (int)op->excl_us_by_n.size())
//         {
//             float v = op->excl_us_by_n[op->opt_tpc_exclusive];
//             if (std::isfinite(v) && v > 0.0f) us = v;
//         }

//         if ((!std::isfinite(us) || us <= 0.0f) && op->duration > 0.0f) {
//             // op.duration is typically ms in your pipeline
//             us = op->duration * 1000.0f;
//         }

//         if (!std::isfinite(us) || us <= 0.0f) us = 1000.0f; // 1ms fallback
//         return us;
//     };

//     auto mark_kernel_running = [&](int cid, op_info* op, int tpc_raw) {
//         if (!op) return;

//         const int tpc_used = normalize_tpc_used(tpc_raw);
//         const float excl_us = estimate_excl_us(op, tpc_used);
//         const uint64_t st = now_ns();
//         const uint64_t dur_ns = (uint64_t)std::llround((double)excl_us * 1.2 * 1000.0); // us -> ns
//         const uint64_t fn = st + dur_ns;

//         op->tpc_used      = tpc_used;
//         op->is_running    = true;
//         op->est_start_ns  = st;
//         op->est_finish_ns = fn;

//         auto it = std::find_if(g_running_ops.begin(), g_running_ops.end(),
//             [&](const RunningOp& rk) {
//                 return rk.client_id == cid && rk.kernel_id == op->id;
//             });

//         if (it == g_running_ops.end()) {
//             g_running_ops.push_back(RunningOp{cid, op->id, op});
//         } else {
//             it->op = op; // refresh pointer
//         }
//     };

//     auto popcount_mask = [&](uint32_t x) -> int {
//         uint32_t used_bits = (HW_TPCS == 32) ? 0xFFFFFFFFu : ((1u << HW_TPCS) - 1u);
//         return __builtin_popcount(x & used_bits);
//     };

//     auto shrink_mask_for_client = [&](int idx, int needed_tpcs) {
//         using tpc_mask_t = uint32_t;
//         tpc_mask_t owned = localMask[idx];
//         int have = popcount_mask(owned);

//         if (have <= needed_tpcs) return;

//         int to_release = have - needed_tpcs;
//         for (int i = HW_TPCS - 1; i >= 0 && to_release > 0; --i) {
//             if ((owned >> i) & tpc_mask_t(1)) {
//                 owned &= ~(tpc_mask_t(1) << i);

//                 // keep consistent with your mask semantics
//                 mask &= ~(tpc_mask_t(1) << i);

//                 if (tpc_usage_count[i] > 0) --tpc_usage_count[i];
//                 ++num_tpcs;  // assumes num_tpcs is "free tpcs"
//                 --to_release;
//             }
//         }
//         localMask[idx] = owned;
//     };

//     auto update_and_build_running_ops = [&](uint64_t now, int exclude_client_id,
//                                             std::vector<const op_info*>& out_running)
//     {
//         out_running.clear();
//         out_running.reserve(g_running_ops.size());

//         std::vector<RunningOp> new_vec;
//         new_vec.reserve(g_running_ops.size());

//         std::vector<int> needed_tpcs(num_clients, 0);

//         auto is_finished = [&](const RunningOp& rk) -> bool {
//             op_info* op = rk.op;
//             if (!op) return true;
//             if (!op->is_running) return true;
//             if (op->est_finish_ns > 0 && op->est_finish_ns <= now) return true;
//             return false;
//         };

//         for (auto& rk : g_running_ops) {
//             if (is_finished(rk)) {
//                 if (rk.op) {
//                     rk.op->is_running    = false;
//                     rk.op->est_start_ns  = 0;
//                     rk.op->est_finish_ns = 0;
//                 }
//                 continue;
//             }

//             new_vec.push_back(rk);

//             if (rk.client_id >= 0 && rk.client_id < num_clients && rk.op) {
//                 needed_tpcs[rk.client_id] =
//                     std::max(needed_tpcs[rk.client_id], normalize_tpc_used(rk.op->tpc_used));
//             }

//             if (rk.client_id != exclude_client_id && rk.op) {
//                 out_running.push_back(rk.op);
//             }
//         }

//         // apply unset/shrink once per client
//         for (int cid = 0; cid < num_clients; ++cid) {
//             int need = needed_tpcs[cid];
//             if (need <= 0) {
//                 unsetmask(cid);
//             } else {
//                 shrink_mask_for_client(cid, need);
//             }
//         }

//         g_running_ops.swap(new_vec);
//     };

//     auto clear_running_ops_for_client = [&](int cid) {
//         for (auto& rk : g_running_ops) {
//             if (rk.client_id == cid && rk.op) {
//                 rk.op->is_running    = false;
//                 rk.op->est_start_ns  = 0;
//                 rk.op->est_finish_ns = 0;
//             }
//         }
//         g_running_ops.erase(
//             std::remove_if(g_running_ops.begin(), g_running_ops.end(),
//                 [&](const RunningOp& rk) { return rk.client_id == cid; }),
//             g_running_ops.end());
//     };

//     while (1) {

//         // Update running state every loop so finished kernels are always reaped
//         if (!warmup) {
//             std::vector<const op_info*> running_now;
//             update_and_build_running_ops(now_ns(), -1, running_now);
//         }

//         std::vector<func_record*> frecords(num_clients, NULL);

//         for (int i = 0; i < num_clients; i++) {

//             if (is_executing[i] == true) continue;
//             if (seen[i] == num_client_kernels[i]) continue;

//             pthread_mutex_lock(client_mutexes[i]);
//             volatile int sz = client_buffers[i]->size();
//             if (sz > 0) {
//                 frecords[i] = &(client_buffers[i]->front());
//                 int cur_iter = num_client_cur_iters[i];
//                 if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
//                     client_starts[i] = std::chrono::high_resolution_clock::now();
//                     client_starts_set[i][cur_iter] = true;
//                 }
//             }
//             pthread_mutex_unlock(client_mutexes[i]);
//         }

//         int num_all_clients = num_clients;

//         // Ready compute clients are selected by priority queue
//         std::priority_queue<Client, std::vector<Client>, ClientPriorityLess> ready_pq;

//         for (int i = 0; i < num_clients; i++) {
//             if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
//                 (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
//                 (stop_ack[i] == true))
//             {
//                 num_all_clients -= 1;
//             }
//         }

//         for (int j = 0; j < num_clients; ++j) {
//             if (frecords[j] != NULL) {
//                 const bool is_non_compute =
//                     (frecords[j]->type == MALLOC_RECORD ||
//                      frecords[j]->type == MEMCPY_RECORD ||
//                      frecords[j]->type == MEMSET_RECORD ||
//                      frecords[j]->type == FREE_RECORD);

//                 if (!is_non_compute && !warmup) {
//                     ready_pq.emplace(j, num_client_cur_iters[j], seen[j], num_client_kernels[j]);
//                 }
//                 else {
//                     schedule_kernel(*(frecords[j]), sched_streams[j], j,
//                                     events[j][event_ids[j]], seen, event_ids, j);
//                     pop_from_queue(client_buffers[j], client_mutexes[j], j);
//                 }
//             }
//         }

//         // ------------------------------------------------------------
//         // Compute scheduling:
//         //   - extract by priority
//         //   - build long/short queues
//         //   - long first
//         //   - short after long
//         // ------------------------------------------------------------
//         if (!warmup && !ready_pq.empty()) {

//             std::vector<int> long_q;
//             std::vector<int> short_q;
//             long_q.reserve(num_clients);
//             short_q.reserve(num_clients);

//             while (!ready_pq.empty()) {
//                 Client c = ready_pq.top();
//                 ready_pq.pop();

//                 const int cid = c.id;
//                 const op_info& op = op_info_vector[cid][seen[cid]];
//                 if (op.is_short == 1) short_q.push_back(cid);
//                 else if (op.is_short == 0) long_q.push_back(cid);
//             }

//             // -------------------------
//             // 1) LONG first
//             // -------------------------
//             // if (long_q.size() > 3) {
//             if (long_q.size() > 1) {

//                 // take top-2 ready longs (already in priority order)
//                 const int pair_cids[2] = { long_q[0], long_q[1] };

//                 op_info* pair_ops[2] = {
//                     &op_info_vector[pair_cids[0]][seen[pair_cids[0]]],
//                     &op_info_vector[pair_cids[1]][seen[pair_cids[1]]]
//                 };

//                 int   best_tA = -1, best_tB = -1;
//                 float best_pred = 0.0f, best_mult = 1.0f, best_exclA = NAN, best_exclB = NAN;

//                 if (!warmup) dbg_cnt++;

//                 if (DBG_OK()) {
//                     const int cidA = pair_cids[0];
//                     const int cidB = pair_cids[1];
//                     const op_info& opA = *pair_ops[0];
//                     const op_info& opB = *pair_ops[1];

//                     printf("[FICKS-DEC] decision #%ld  iterA=%d iterB=%d  "
//                            "cidA=%d seenA=%d (id=%d,name=%s,cl=%d)  "
//                            "cidB=%d seenB=%d (id=%d,name=%s,cl=%d)\n",
//                            dbg_cnt,
//                            num_client_cur_iters[cidA], num_client_cur_iters[cidB],
//                            cidA, seen[cidA], opA.id, opA.name.c_str(), opA.cluster,
//                            cidB, seen[cidB], opB.id, opB.name.c_str(), opB.cluster);
//                     fflush(stdout);
//                 }

//                 pick_best_split_for_two(pair_cids[0], pair_cids[1],
//                                         *pair_ops[0], *pair_ops[1],
//                                         best_tA, best_tB,
//                                         best_pred, best_mult, best_exclA, best_exclB);

//                 if (DBG_OK()) {
//                     const char* mode = (best_tA < 0 && best_tB < 0) ? "NO_MASK" : "MASKED";
//                     printf("[FICKS-DEC] CHOSEN  mode=%s  split=(%d,%d)  mult=%.6f  "
//                            "exclA=%.3f  exclB=%.3f  pred_makespan=%.3f\n",
//                            mode, best_tA, best_tB, (double)best_mult,
//                            (double)best_exclA, (double)best_exclB, (double)best_pred);
//                     fflush(stdout);
//                 }

//                 const int best_t[2] = { best_tA, best_tB };

//                 for (int p = 0; p < 2; ++p) {
//                     const int cid = pair_cids[p];
//                     op_info* op_ptr = &op_info_vector[cid][seen[cid]]; // capture before schedule

//                     if (best_t[p] > 0) {
//                         setmask_O(best_t[p], cid);
//                     } else {
//                         setmask_O(24, cid); // no-mask mode: full device
//                     }

//                     schedule_one(cid, frecords);

//                     // push running op (long)
//                     mark_kernel_running(cid, op_ptr, (best_t[p] > 0 ? best_t[p] : 24));
//                 }

//                 // for (int cid : long_q) {
//                 //     op_info* op_ptr = &op_info_vector[cid][seen[cid]]; // capture before schedule

//                 //     schedule_kernel(*(frecords[cid]),
//                 //                     sched_streams[cid],
//                 //                     cid,
//                 //                     events[cid][event_ids[cid]],
//                 //                     seen, event_ids, cid);
//                 //     pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);

//                 //     // push running op (short too)
//                 //     mark_kernel_running(cid, op_ptr, 24);
//                 // }

//             }
//             else if (long_q.size() == 1) {
//             // else if (long_q.size() > 0 && long_q.size() < 3) {   // effectively size == 2

//                 // mark all cids already in long_q
//                 std::vector<char> in_long_q(num_clients, 0);
//                 for (int x : long_q) {
//                     if (x >= 0 && x < num_clients) in_long_q[x] = 1;
//                 }

//                 auto launch_long = [&](int cid) {
//                     op_info* op_ptr = &op_info_vector[cid][seen[cid]]; // capture before schedule

//                     setmask_O(24, cid);
//                     schedule_kernel(*(frecords[cid]),
//                                     sched_streams[cid],
//                                     cid,
//                                     events[cid][event_ids[cid]],
//                                     seen, event_ids, cid);
//                     pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);

//                     mark_kernel_running(cid, op_ptr, 24);
//                 };

//                 // deadlock guard: check active clients NOT in long_q
//                 bool other_has_work = false;
//                 for (int j = 0; j < num_clients; ++j) {
//                     if (in_long_q[j]) continue; // skip all cids in long_q

//                     if ((num_client_cur_iters[j] == num_client_max_iters[j]) ||
//                         (warmup && (num_client_cur_iters[j] == warmup_iters)) ||
//                         (stop_ack[j] == true))
//                         continue;

//                     if (seen[j] < num_client_kernels[j]) {
//                         other_has_work = true;
//                         break;
//                     }
//                 }

//                 if (!other_has_work) {
//                     // launch all long kernels in this round
//                     for (int cid : long_q) {
//                         launch_long(cid);
//                     }
//                 } else {
//                     int look = depth;
//                     if (look < 0) look = 0;

//                     bool found_other_long_within_depth = false;

//                     if (look > 0) {
//                         for (int j = 0; j < num_clients; ++j) {
//                             if (in_long_q[j]) continue; // only search cids not in long_q

//                             if ((num_client_cur_iters[j] == num_client_max_iters[j]) ||
//                                 (warmup && (num_client_cur_iters[j] == warmup_iters)) ||
//                                 (stop_ack[j] == true))
//                                 continue;

//                             if (seen[j] >= num_client_kernels[j]) continue;
                            
//                             if (seen[j] < 1) continue;

//                             for (int k = 0; k < look; ++k) {
//                                 int idx = seen[j] + k;
//                                 if (idx >= num_client_kernels[j]) break;

//                                 const op_info& opj = op_info_vector[j][idx];
//                                 if (opj.is_short == 0) {
//                                     found_other_long_within_depth = true;
//                                     break;
//                                 }
//                             }

//                             if (found_other_long_within_depth) break;
//                         }
//                     }

//                     if (!found_other_long_within_depth) {
//                         // no useful partner incoming soon -> launch all in long_q now
//                         for (int cid : long_q) {
//                             launch_long(cid);
//                         }
//                     }
//                 }
//             }


//             // -------------------------
//             // 2) SHORT after long
//             // -------------------------
//             if (!short_q.empty()) {
//                 for (int cid : short_q) {
//                     op_info* op_ptr = &op_info_vector[cid][seen[cid]]; // capture before schedule

//                     setmask_O(24, cid);
//                     schedule_kernel(*(frecords[cid]),
//                                     sched_streams[cid],
//                                     cid,
//                                     events[cid][event_ids[cid]],
//                                     seen, event_ids, cid);
//                     pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);

//                     // push running op (short too)
//                     mark_kernel_running(cid, op_ptr, 24);
//                 }
//             }
//         }

//         int finished = 0;
//         for (int i = 0; i < num_clients; i++) {

//             if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
//                 (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
//                 (stop_ack[i] == true))
//             {
//                 finished += 1;
//             }
//             else if (seen[i] == num_client_kernels[i]) {
//                 if (!locked[i]) {
//                     pthread_mutex_lock(client_mutexes[i]);
//                     locked[i] = true;
//                     DEBUG_PRINT("LOCK CLIENT %d\n", i);
//                 }

//                 bool ready = true;
//                 if (seq) {
//                     if (event_ids[0] >= 1) {
//                         if (cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess)
//                             ready &= false;
//                     }
//                 }
//                 else {
//                     if (event_ids[i] >= 1) {
//                         if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess)
//                             ready &= false;
//                     }
//                 }

//                 if (ready) {
//                     unsetmask(i);
//                     clear_running_ops_for_client(i);

//                     seen[i] = 0;
//                     if (seq) event_ids[0] = 0;
//                     event_ids[i] = 0;
//                     streams[i] = -1;
//                     fidx[i] = 0;
//                     request_status[i][num_client_cur_iters[i]] = true;

//                     pthread_mutex_unlock(client_mutexes[i]);
//                     num_client_cur_iters[i] += 1;
//                     locked[i] = false;
//                     client_progress[i] = 0;

//                     auto end = std::chrono::high_resolution_clock::now();
//                     float duration = std::chrono::duration_cast<std::chrono::microseconds>(
//                                          end - client_starts[i]).count();
//                     duration /= 1000.0f;
//                     client_durations[i].push_back(duration);
//                 }
//             }
//         }

//         if (finished == num_clients) break;
//     }

//     if (!warmup) {
//         auto end_total = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
//                             end_total - start_total).count();
//         duration /= 1000.0;
//         printf("Total loop took %ld nanoseconds\n", duration);
//     }

//     return NULL;
// }




void* Scheduler::busy_wait_ficks(int num_clients, int iter, bool warmup, int warmup_iters,
                                 bool seq, int depth, int hp_limit, int update_start)
{
    DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);

    depth = 3;
    // depth = g_TD1;

    printf("depth is %d\n", depth);
    using Clock = std::chrono::high_resolution_clock;
    auto start_total = Clock::now();

    // ============================================================
    // Decision debug (prints only a bounded number of decisions)
    // ============================================================
    const bool DBG_DECISIONS = false;   // flip to false to silence
    const int  DBG_EVERY     = 1;       // print every N decisions
    const long DBG_MAX       = 2000;    // hard cap to avoid log storms
    static long dbg_cnt = 0;

    auto DBG_OK = [&]() -> bool {
        if (!DBG_DECISIONS) return false;
        if (warmup) return false;
        if (dbg_cnt >= DBG_MAX) return false;
        return ((dbg_cnt % DBG_EVERY) == 0);
    };

    auto print_split = [&](int cidA, int cidB,
                           const op_info& opA, const op_info& opB,
                           int tA, int tB, bool ok, float mult,
                           float exclA, float exclB, float pred,
                           bool is_best)
    {
        if (!DBG_OK()) return;

        // note: tA/tB == -1 means no mask
        printf("[FICKS-DEC] cand  cidA=%d(kidx=%d,id=%d,name=%s,cl=%d)  "
               "cidB=%d(kidx=%d,id=%d,name=%s,cl=%d)  "
               "split=(%d,%d)  ok=%d  mult=%.6f  exclA=%.3f  exclB=%.3f  "
               "max=%.3f  pred=%.3f%s\n",
               cidA, (int)(opA.id), opA.id, opA.name.c_str(), opA.cluster,
               cidB, (int)(opB.id), opB.id, opB.name.c_str(), opB.cluster,
               tA, tB, ok ? 1 : 0, (double)mult,
               (double)exclA, (double)exclB, (double)std::max(exclA, exclB),
               (double)pred,
               is_best ? "  <-- best" : "");
        fflush(stdout);
    };

    auto schedule_one = [&](int cid, std::vector<func_record*>& frecords) {
        schedule_kernel(*(frecords[cid]),
                        sched_streams[cid],
                        cid,
                        events[cid][event_ids[cid]],
                        seen, event_ids, cid);
        pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);
    };

    auto pick_best_split_for_two = [&](int cidA, int cidB,
                                   op_info& opA, op_info& opB,
                                   int& out_tA, int& out_tB,
                                   float& out_best_pred,
                                   float& out_best_mult,
                                   float& out_best_exclA,
                                   float& out_best_exclB)
    {
        const int ca = opA.cluster;
        const int cb = opB.cluster;

        std::vector<std::pair<int,int>> splits;
        build_candidate_splits(splits);

        float best_pred  = std::numeric_limits<float>::infinity();
        int   best_tA    = -1;
        int   best_tB    = -1;
        float best_mult  = 1.0f;
        float best_exclA = NAN;
        float best_exclB = NAN;

        for (const auto& sp : splits) {
            int tA = sp.first;
            int tB = sp.second;

            float mult = 1.0f;
            bool ok = lookup_split_mult(ca, cb, tA, tB, mult);

            // allow (-1,-1) even if LUT missing
            if (!ok && !(tA == -1 && tB == -1)) continue;

            float exclA = excl_us_at_tpc_or_full(opA, tA);
            float exclB = excl_us_at_tpc_or_full(opB, tB);
            float pred  = mult * std::max(exclA, exclB);

            bool is_best = false;
            if (pred < best_pred) {
                best_pred  = pred;
                best_tA    = tA;
                best_tB    = tB;
                best_mult  = mult;
                best_exclA = exclA;
                best_exclB = exclB;
                is_best    = true;
            }

            print_split(cidA, cidB, opA, opB, tA, tB, ok, mult, exclA, exclB, pred, is_best);
        }

        // safe fallback
        if (!std::isfinite(best_pred)) {
            best_tA    = -1;
            best_tB    = -1;
            best_mult  = 1.0f;
            best_exclA = excl_us_at_tpc_or_full(opA, -1);
            best_exclB = excl_us_at_tpc_or_full(opB, -1);
            best_pred  = best_mult * std::max(best_exclA, best_exclB);
        }

        // NEW RULE:
        // if chosen split is "too close" (difference <= 4), use no-mask for both
        // (only apply when both are real masked tpc values)
        constexpr int kSplitDiffToNoMask = 4;
        if (best_tA > 0 && best_tB > 0 && std::abs(best_tA - best_tB) <= kSplitDiffToNoMask) {
            float nm_mult = 1.0f;
            bool nm_ok = lookup_split_mult(ca, cb, -1, -1, nm_mult);
            if (!nm_ok) nm_mult = 1.0f; // keep current behavior

            best_tA    = -1;
            best_tB    = -1;
            best_mult  = nm_mult;
            best_exclA = excl_us_at_tpc_or_full(opA, -1);
            best_exclB = excl_us_at_tpc_or_full(opB, -1);
            best_pred  = best_mult * std::max(best_exclA, best_exclB);

            if (DBG_OK()) {
                printf("[FICKS-DEC] override: |tA-tB|<=%d => force NO_MASK (-1,-1)\n",
                    kSplitDiffToNoMask);
                fflush(stdout);
            }
        }

        out_tA         = best_tA;
        out_tB         = best_tB;
        out_best_pred  = best_pred;
        out_best_mult  = best_mult;
        out_best_exclA = best_exclA;
        out_best_exclB = best_exclB;
    };

    // ============================================================
    // Running-ops bookkeeping (push BOTH long/short; no-mask => all TPC)
    // ============================================================
    const int HW_TPCS = 24; // change if needed on non-A4000

    auto now_ns = [&]() -> uint64_t {
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count();
    };

    auto normalize_tpc_used = [&](int tpc_raw) -> int {
        if (tpc_raw <= 0 || tpc_raw > HW_TPCS) return HW_TPCS; // no-mask mode uses all TPC
        return tpc_raw;
    };

    auto estimate_excl_us = [&](const op_info* op, int tpc_used_norm) -> float {
        if (!op) return 1000.0f;

        float us = NAN;
        if (tpc_used_norm >= 0 && tpc_used_norm < (int)op->excl_us_by_n.size()) {
            float v = op->excl_us_by_n[tpc_used_norm];
            if (std::isfinite(v) && v > 0.0f) us = v;
        }

        if (!std::isfinite(us)) {
            us = excl_us_at_tpc_or_full(*op, tpc_used_norm);
        }

        if ((!std::isfinite(us) || us <= 0.0f) && op->opt_tpc_exclusive > 0 &&
            op->opt_tpc_exclusive < (int)op->excl_us_by_n.size())
        {
            float v = op->excl_us_by_n[op->opt_tpc_exclusive];
            if (std::isfinite(v) && v > 0.0f) us = v;
        }

        if ((!std::isfinite(us) || us <= 0.0f) && op->duration > 0.0f) {
            // op.duration is typically ms in your pipeline
            us = op->duration * 1000.0f;
        }

        if (!std::isfinite(us) || us <= 0.0f) us = 1000.0f; // 1ms fallback
        return us;
    };

    auto mark_kernel_running = [&](int cid, op_info* op, int tpc_raw) {
        if (!op) return;

        const int tpc_used = normalize_tpc_used(tpc_raw);
        const float excl_us = estimate_excl_us(op, tpc_used);
        const uint64_t st = now_ns();
        const uint64_t dur_ns = (uint64_t)std::llround((double)excl_us * 1.2 * 1000.0); // us -> ns
        const uint64_t fn = st + dur_ns;

        op->tpc_used      = tpc_used;
        op->is_running    = true;
        op->est_start_ns  = st;
        op->est_finish_ns = fn;

        auto it = std::find_if(g_running_ops.begin(), g_running_ops.end(),
            [&](const RunningOp& rk) {
                return rk.client_id == cid && rk.kernel_id == op->id;
            });

        if (it == g_running_ops.end()) {
            g_running_ops.push_back(RunningOp{cid, op->id, op});
        } else {
            it->op = op; // refresh pointer
        }
    };

    auto popcount_mask = [&](uint32_t x) -> int {
        uint32_t used_bits = (HW_TPCS == 32) ? 0xFFFFFFFFu : ((1u << HW_TPCS) - 1u);
        return __builtin_popcount(x & used_bits);
    };

    auto shrink_mask_for_client = [&](int idx, int needed_tpcs) {
        using tpc_mask_t = uint32_t;
        tpc_mask_t owned = localMask[idx];
        int have = popcount_mask(owned);

        if (have <= needed_tpcs) return;

        int to_release = have - needed_tpcs;
        for (int i = HW_TPCS - 1; i >= 0 && to_release > 0; --i) {
            if ((owned >> i) & tpc_mask_t(1)) {
                owned &= ~(tpc_mask_t(1) << i);

                // keep consistent with your mask semantics
                mask &= ~(tpc_mask_t(1) << i);

                if (tpc_usage_count[i] > 0) --tpc_usage_count[i];
                ++num_tpcs;  // assumes num_tpcs is "free tpcs"
                --to_release;
            }
        }
        localMask[idx] = owned;
    };

    auto update_and_build_running_ops = [&](uint64_t now, int exclude_client_id,
                                            std::vector<const op_info*>& out_running)
    {
        out_running.clear();
        out_running.reserve(g_running_ops.size());

        std::vector<RunningOp> new_vec;
        new_vec.reserve(g_running_ops.size());

        std::vector<int> needed_tpcs(num_clients, 0);

        auto is_finished = [&](const RunningOp& rk) -> bool {
            op_info* op = rk.op;
            if (!op) return true;
            if (!op->is_running) return true;
            if (op->est_finish_ns > 0 && op->est_finish_ns <= now) return true;
            return false;
        };

        for (auto& rk : g_running_ops) {
            if (is_finished(rk)) {
                if (rk.op) {
                    rk.op->is_running    = false;
                    rk.op->est_start_ns  = 0;
                    rk.op->est_finish_ns = 0;
                }
                continue;
            }

            new_vec.push_back(rk);

            if (rk.client_id >= 0 && rk.client_id < num_clients && rk.op) {
                needed_tpcs[rk.client_id] =
                    std::max(needed_tpcs[rk.client_id], normalize_tpc_used(rk.op->tpc_used));
            }

            if (rk.client_id != exclude_client_id && rk.op) {
                out_running.push_back(rk.op);
            }
        }

        // apply unset/shrink once per client
        for (int cid = 0; cid < num_clients; ++cid) {
            int need = needed_tpcs[cid];
            if (need <= 0) {
                unsetmask(cid);
            } else {
                shrink_mask_for_client(cid, need);
            }
        }

        g_running_ops.swap(new_vec);
    };

    auto clear_running_ops_for_client = [&](int cid) {
        for (auto& rk : g_running_ops) {
            if (rk.client_id == cid && rk.op) {
                rk.op->is_running    = false;
                rk.op->est_start_ns  = 0;
                rk.op->est_finish_ns = 0;
            }
        }
        g_running_ops.erase(
            std::remove_if(g_running_ops.begin(), g_running_ops.end(),
                [&](const RunningOp& rk) { return rk.client_id == cid; }),
            g_running_ops.end());
    };

    while (1) {

        // Update running state every loop so finished kernels are always reaped
        if (!warmup) {
            std::vector<const op_info*> running_now;
            update_and_build_running_ops(now_ns(), -1, running_now);
        }

        std::vector<func_record*> frecords(num_clients, NULL);

        for (int i = 0; i < num_clients; i++) {

            if (is_executing[i] == true) continue;
            if (seen[i] == num_client_kernels[i]) continue;

            pthread_mutex_lock(client_mutexes[i]);
            volatile int sz = client_buffers[i]->size();
            if (sz > 0) {
                frecords[i] = &(client_buffers[i]->front());
                int cur_iter = num_client_cur_iters[i];
                if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
                    client_starts[i] = std::chrono::high_resolution_clock::now();
                    client_starts_set[i][cur_iter] = true;
                }
            }
            pthread_mutex_unlock(client_mutexes[i]);
        }

        int num_all_clients = num_clients;

        // Ready compute clients are selected by priority queue
        std::priority_queue<Client, std::vector<Client>, ClientPriorityLess> ready_pq;

        for (int i = 0; i < num_clients; i++) {
            if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
                (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
                (stop_ack[i] == true))
            {
                num_all_clients -= 1;
            }
        }

        for (int j = 0; j < num_clients; ++j) {
            if (frecords[j] != NULL) {
                const bool is_non_compute =
                    (frecords[j]->type == MALLOC_RECORD ||
                     frecords[j]->type == MEMCPY_RECORD ||
                     frecords[j]->type == MEMSET_RECORD ||
                     frecords[j]->type == FREE_RECORD);

                if (!is_non_compute && !warmup) {
                    ready_pq.emplace(j, num_client_cur_iters[j], seen[j], num_client_kernels[j]);
                }
                else {
                    schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                    events[j][event_ids[j]], seen, event_ids, j);
                    pop_from_queue(client_buffers[j], client_mutexes[j], j);
                }
            }
        }

        // ------------------------------------------------------------
        // Compute scheduling:
        //   - extract by priority
        //   - build long/short queues
        //   - long first
        //   - short after long
        // ------------------------------------------------------------
        if (!warmup && !ready_pq.empty()) {

            std::vector<int> long_q;
            std::vector<int> short_q;
            long_q.reserve(num_clients);
            short_q.reserve(num_clients);

            while (!ready_pq.empty()) {
                Client c = ready_pq.top();
                ready_pq.pop();

                const int cid = c.id;
                const op_info& op = op_info_vector[cid][seen[cid]];
                if (op.is_short == 1) short_q.push_back(cid);
                else if (op.is_short == 0) long_q.push_back(cid);
            }

            // -------------------------
            // 1) LONG first
            // -------------------------
            // if (long_q.size() > 3) {
            if (long_q.size() > 1) {

                // take top-2 ready longs (already in priority order)
                const int pair_cids[2] = { long_q[0], long_q[1] };

                op_info* pair_ops[2] = {
                    &op_info_vector[pair_cids[0]][seen[pair_cids[0]]],
                    &op_info_vector[pair_cids[1]][seen[pair_cids[1]]]
                };

                int   best_tA = -1, best_tB = -1;
                float best_pred = 0.0f, best_mult = 1.0f, best_exclA = NAN, best_exclB = NAN;

                if (!warmup) dbg_cnt++;

                if (DBG_OK()) {
                    const int cidA = pair_cids[0];
                    const int cidB = pair_cids[1];
                    const op_info& opA = *pair_ops[0];
                    const op_info& opB = *pair_ops[1];

                    printf("[FICKS-DEC] decision #%ld  iterA=%d iterB=%d  "
                           "cidA=%d seenA=%d (id=%d,name=%s,cl=%d)  "
                           "cidB=%d seenB=%d (id=%d,name=%s,cl=%d)\n",
                           dbg_cnt,
                           num_client_cur_iters[cidA], num_client_cur_iters[cidB],
                           cidA, seen[cidA], opA.id, opA.name.c_str(), opA.cluster,
                           cidB, seen[cidB], opB.id, opB.name.c_str(), opB.cluster);
                    fflush(stdout);
                }

                pick_best_split_for_two(pair_cids[0], pair_cids[1],
                                        *pair_ops[0], *pair_ops[1],
                                        best_tA, best_tB,
                                        best_pred, best_mult, best_exclA, best_exclB);

                if (DBG_OK()) {
                    const char* mode = (best_tA < 0 && best_tB < 0) ? "NO_MASK" : "MASKED";
                    printf("[FICKS-DEC] CHOSEN  mode=%s  split=(%d,%d)  mult=%.6f  "
                           "exclA=%.3f  exclB=%.3f  pred_makespan=%.3f\n",
                           mode, best_tA, best_tB, (double)best_mult,
                           (double)best_exclA, (double)best_exclB, (double)best_pred);
                    fflush(stdout);
                }

                const int best_t[2] = { best_tA, best_tB };

                for (int p = 0; p < 2; ++p) {
                    const int cid = pair_cids[p];
                    op_info* op_ptr = &op_info_vector[cid][seen[cid]]; // capture before schedule

                    if (best_t[p] > 0) {
                        setmask_O(best_t[p], cid);
                    } else {
                        setmask_O(24, cid); // no-mask mode: full device
                    }

                    schedule_one(cid, frecords);

                    // push running op (long)
                    mark_kernel_running(cid, op_ptr, (best_t[p] > 0 ? best_t[p] : 24));
                }

                // for (int cid : long_q) {
                //     op_info* op_ptr = &op_info_vector[cid][seen[cid]]; // capture before schedule

                //     schedule_kernel(*(frecords[cid]),
                //                     sched_streams[cid],
                //                     cid,
                //                     events[cid][event_ids[cid]],
                //                     seen, event_ids, cid);
                //     pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);

                //     // push running op (short too)
                //     mark_kernel_running(cid, op_ptr, 24);
                // }

            }
            else if (long_q.size() == 1) {
            // else if (long_q.size() > 0 && long_q.size() < 3) {   // effectively size == 2

                // mark all cids already in long_q
                std::vector<char> in_long_q(num_clients, 0);
                for (int x : long_q) {
                    if (x >= 0 && x < num_clients) in_long_q[x] = 1;
                }

                auto launch_long = [&](int cid) {
                    op_info* op_ptr = &op_info_vector[cid][seen[cid]]; // capture before schedule

                    setmask_O(op_ptr->opt_tpc_colocated, cid);
                    schedule_kernel(*(frecords[cid]),
                                    sched_streams[cid],
                                    cid,
                                    events[cid][event_ids[cid]],
                                    seen, event_ids, cid);
                    pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);

                    mark_kernel_running(cid, op_ptr, op_ptr->opt_tpc_colocated);
                };

                // deadlock guard: check active clients NOT in long_q
                bool other_has_work = false;
                for (int j = 0; j < num_clients; ++j) {
                    if (in_long_q[j]) continue; // skip all cids in long_q

                    if ((num_client_cur_iters[j] == num_client_max_iters[j]) ||
                        (warmup && (num_client_cur_iters[j] == warmup_iters)) ||
                        (stop_ack[j] == true))
                        continue;

                    if (seen[j] < num_client_kernels[j]) {
                        other_has_work = true;
                        break;
                    }
                }

                if (!other_has_work) {
                    // launch all long kernels in this round
                    for (int cid : long_q) {
                        launch_long(cid);
                    }
                } else {
                    int look = depth;
                    if (look < 0) look = 0;

                    bool found_other_long_within_depth = false;

                    if (look > 0) {
                        for (int j = 0; j < num_clients; ++j) {
                            if (in_long_q[j]) continue; // only search cids not in long_q

                            if ((num_client_cur_iters[j] == num_client_max_iters[j]) ||
                                (warmup && (num_client_cur_iters[j] == warmup_iters)) ||
                                (stop_ack[j] == true))
                                continue;

                            if (seen[j] >= num_client_kernels[j]) continue;
                            
                            // if (seen[j] < 1) continue;

                            for (int k = 0; k < look; ++k) {
                                int idx = seen[j] + k;
                                if (idx >= num_client_kernels[j]) break;

                                const op_info& opj = op_info_vector[j][idx];
                                if (opj.is_short == 0) {
                                    found_other_long_within_depth = true;
                                    break;
                                }
                            }

                            if (found_other_long_within_depth) break;
                        }
                    }

                    if (!found_other_long_within_depth) {
                        // no useful partner incoming soon -> launch all in long_q now
                        for (int cid : long_q) {
                            launch_long(cid);
                        }
                    }
                }
            }


            // -------------------------
            // 2) SHORT after long
            // -------------------------
            if (!short_q.empty()) {
                for (int cid : short_q) {
                    op_info* op_ptr = &op_info_vector[cid][seen[cid]]; // capture before schedule

                    setmask_O(op_ptr->opt_tpc_colocated, cid);
                    schedule_kernel(*(frecords[cid]),
                                    sched_streams[cid],
                                    cid,
                                    events[cid][event_ids[cid]],
                                    seen, event_ids, cid);
                    pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);

                    // push running op (short too)
                    mark_kernel_running(cid, op_ptr, op_ptr->opt_tpc_colocated);
                }
            }
        }

        int finished = 0;
        for (int i = 0; i < num_clients; i++) {

            if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
                (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
                (stop_ack[i] == true))
            {
                finished += 1;
            }
            else if (seen[i] == num_client_kernels[i]) {
                if (!locked[i]) {
                    pthread_mutex_lock(client_mutexes[i]);
                    locked[i] = true;
                    DEBUG_PRINT("LOCK CLIENT %d\n", i);
                }

                bool ready = true;
                if (seq) {
                    if (event_ids[0] >= 1) {
                        if (cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess)
                            ready &= false;
                    }
                }
                else {
                    if (event_ids[i] >= 1) {
                        if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess)
                            ready &= false;
                    }
                }

                if (ready) {
                    unsetmask(i);
                    clear_running_ops_for_client(i);

                    seen[i] = 0;
                    if (seq) event_ids[0] = 0;
                    event_ids[i] = 0;
                    streams[i] = -1;
                    fidx[i] = 0;
                    request_status[i][num_client_cur_iters[i]] = true;

                    pthread_mutex_unlock(client_mutexes[i]);
                    num_client_cur_iters[i] += 1;
                    locked[i] = false;
                    client_progress[i] = 0;

                    auto end = std::chrono::high_resolution_clock::now();
                    float duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                         end - client_starts[i]).count();
                    duration /= 1000.0f;
                    client_durations[i].push_back(duration);
                }
            }
        }

        if (finished == num_clients) break;
    }

    if (!warmup) {
        auto end_total = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_total - start_total).count();
        duration /= 1000.0;
        printf("Total loop took %ld nanoseconds\n", duration);
    }

    return NULL;
}








namespace fs = std::filesystem;
void* Scheduler::busy_wait_profile_co(int num_clients, int iter, bool warmup, int warmup_iters,
                                      bool seq, int depth, int hp_limit, int update_start)
{
    DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);

    // ────────────────────────────────────────────────────────────────────────
    // 0) Resolve model names and filesystem-friendly names
    // ────────────────────────────────────────────────────────────────────────
    std::string model0 = (model_names.size() > 0 ? model_names[0] : std::string("model0"));
    std::string model1 = (model_names.size() > 1 ? model_names[1] : std::string("model1"));

    auto normalize_for_fs = [](std::string s) {
        return s;
    };

    std::string model0_fs = normalize_for_fs(model0);
    std::string model1_fs = normalize_for_fs(model1);

    // Root directory where pair_files, partition_files, pair_durations live
    const std::string PROFILE_ROOT =
        "/home/zixi/orion_bu/artifact_evaluation/FICKS/critical_schedule_profile";

    const std::string PROFILE_PLAN_DIR = PROFILE_ROOT + "/partition_files";
    const std::string DURATION_OUT_DIR = PROFILE_ROOT + "/pair_durations";

    // Ensure output directory exists
    try {
        fs::create_directories(DURATION_OUT_DIR);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] Failed to create directory " << DURATION_OUT_DIR
                  << ": " << e.what() << std::endl;
    }

    // Partition CSV is generated per model-pair:
    //   partition_files/<model0>_<model1>_partitions.csv
    std::string profile_csv =
        PROFILE_PLAN_DIR + "/" + model0_fs + "_" + model1_fs + "_partitions.csv";

    std::cout << "[INFO] Using profile plan CSV: " << profile_csv << std::endl;

    // CSV output path (one file per (model0, model1))
    std::string out_name = "pair_durations_" + model0_fs + "_" + model1_fs + ".csv";
    std::string out_path = DURATION_OUT_DIR + "/" + out_name;

    // ────────────────────────────────────────────────────────────────────────
    // 1) Load profile plan (pairs of {kernel-id, tpc-mask})
    // ────────────────────────────────────────────────────────────────────────
    auto profile_plan = load_profile_plan(profile_csv);
    if (profile_plan.empty()) {
        std::cerr << "No entries loaded from " << profile_csv << ".\n";
        return NULL; // avoid invalid access below
    }
    auto& pair = profile_plan[0];

    // ────────────────────────────────────────────────────────────────────────
    // 2) Per-window indices and masks (2 clients assumed)
    // ────────────────────────────────────────────────────────────────────────
    int window_start[2];
    int window_end[2];
    int schedule_client[2];

    window_start[0] = pair.first.id;
    window_start[1] = pair.second.id;
    window_end[0]   = window_start[0] + 1;
    window_end[1]   = window_start[1] + 1;

    schedule_client[0] = pair.first.tpc;   // may be -1
    schedule_client[1] = pair.second.tpc;  // may be -1

    // ────────────────────────────────────────────────────────────────────────
    // 3) Timers / bookkeeping
    // ────────────────────────────────────────────────────────────────────────
    auto start_total = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> start_profile;

    std::vector<bool> total_client_set(num_clients, false);

    bool finish_profile[2] = {false, false};
    bool finished_c[2]     = {false, false};  // window kernels completed
    bool printed_pair      = false;           // print-once guard
    bool store_profile     = false;           // record once per pair
    std::chrono::time_point<std::chrono::high_resolution_clock> finish_ts[2];

    // ────────────────────────────────────────────────────────────────────────
    // 4) In-memory record struct (write immediately)
    // ────────────────────────────────────────────────────────────────────────
    struct PairRecord {
        int         iteration;        // iteration index when measured
        int         kid0, kid1;       // window_start ids for the pair
        int         tpc0, tpc1;       // masks used for the pair
        double      makespan_us;     // ONLY makespan in microseconds
        std::string mask_mode;        // "nomask" or "masked"
        std::string model0;           // logical model name for client 0
        std::string model1;           // logical model name for client 1
    };

    auto is_nomask_row = [&](int t0, int t1) -> bool {
        // ONLY (-1,-1) means no masking
        return (t0 == -1 && t1 == -1);
    };

    auto kernel_name_safe = [&](int cid, int kid) -> std::string {
        if (cid < 0 || cid >= (int)op_info_vector.size()) return std::string("NA");
        if (kid < 0 || kid >= (int)op_info_vector[cid].size()) return std::string("NA");
        // assuming op_info has a field `name` that is std::string
        return op_info_vector[cid][kid].name;
    };

    // ────────────────────────────────────────────────────────────────────────
    // 5) Main loop
    // ────────────────────────────────────────────────────────────────────────
    while (1) {
        std::vector<bool> scheduled_in_tick(num_clients, false);

        // (5.1) Peek heads
        std::vector<func_record*> frecords(num_clients, NULL);

        for (int i = 0; i < num_clients; i++) {
            if (is_executing[i]) continue;
            if (seen[i] == num_client_kernels[i]) continue;

            pthread_mutex_lock(client_mutexes[i]);
            volatile int sz = client_buffers[i]->size();
            if (sz > 0) {
                frecords[i] = &(client_buffers[i]->front());
                int cur_iter = num_client_cur_iters[i];
                if (seen[i] == 0 && !client_starts_set[i][cur_iter]) {
                    client_starts[i] = std::chrono::high_resolution_clock::now();
                    client_starts_set[i][cur_iter] = true;
                    if (!total_client_set[i]) {
                        total_client_starts[i] = std::chrono::high_resolution_clock::now();
                        total_client_set[i] = true;
                    }
                }
            }
            pthread_mutex_unlock(client_mutexes[i]);
        }

        // (5.3) Drain non-compute records immediately; collect ready compute
        std::vector<int> ready_client;
        for (int j = num_clients - 1; j > -1; --j) {
            if (!frecords[j]) continue;
            if (frecords[j]->type != MALLOC_RECORD &&
                frecords[j]->type != MEMCPY_RECORD &&
                frecords[j]->type != MEMSET_RECORD &&
                frecords[j]->type != FREE_RECORD &&
                num_client_cur_iters[j] > 9)
            {
                ready_client.push_back(j);
            } else {
                scheduled_in_tick[j] = true;

                schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                events[j][event_ids[j]], seen, event_ids, j);
                pop_from_queue(client_buffers[j], client_mutexes[j], j);
            }
        }

        // (5.4) Per-client throttle; release mask if previous op finished
        if (!warmup) {
            bool canSchedule[num_clients];
            for (int i = 0; i < num_clients; ++i) {
                canSchedule[i] = true;
                if (event_ids[i] >= 1) {
                    if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                        if (!finished_c[i] && seen[i] == window_end[i]) {
                            finished_c[i] = true;
                            finish_ts[i] = std::chrono::high_resolution_clock::now();
                        }
                        unsetmask(i);
                    } else {
                        canSchedule[i] = false;
                    }
                }
            }

            // (5.5) Drive both clients to the window start
            if (!finish_profile[0] && !finish_profile[1] &&
                num_client_cur_iters[0] == num_client_cur_iters[1] &&
                !ready_client.empty())
            {
                for (int client_id : ready_client) {
                    if (seen[client_id] < window_start[client_id]) {
                        if (frecords[client_id] != NULL) {
                            scheduled_in_tick[client_id] = true;

                            schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id,
                                            events[client_id][event_ids[client_id]], seen, event_ids, client_id);
                            pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
                        }
                    }
                }

                // (5.6) Launch exactly one window kernel per client under masks
                bool any_scheduled_this_tick = false;
                for (int c = 0; c < num_clients; ++c) {
                    if (scheduled_in_tick[c]) { any_scheduled_this_tick = true; break; }
                }

                if (seen[0] == window_start[0] && seen[1] == window_start[1] &&
                    !any_scheduled_this_tick)
                {
                    bool begin_sche = true;
                    for (int i = 0; i < 2; ++i) {
                        if ((seen[i] != 0) && seen[i] == window_start[i]) {
                            if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess) {
                                begin_sche = false;
                                break;
                            }
                        }
                    }

                    if (begin_sche && ready_client.size() == 2) {
                        printed_pair = false;
                        finished_c[0] = finished_c[1] = false;
                        store_profile = false;

                        // IMPORTANT: one shared start timestamp for makespan
                        start_profile = std::chrono::high_resolution_clock::now();

                        // ---- (-1,-1) means NO MASKING ----
                        bool nomask = is_nomask_row(schedule_client[0], schedule_client[1]);
                        if (nomask) {
                            unsetmask(0);
                            unsetmask(1);
                        }

                        for (int client_id : ready_client) {
                            if (frecords[client_id] != NULL &&
                                seen[client_id] < window_end[client_id])
                            {
                                if (!nomask) {
                                    setmask(schedule_client[client_id], client_id);
                                }

                                scheduled_in_tick[client_id] = true;

                                schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id,
                                                events[client_id][event_ids[client_id]], seen, event_ids, client_id);
                                pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
                            }
                        }
                    }
                }
            } else {
                // (5.7) After window kernel(s), continue normal scheduling for finished clients
                bool begin_sche = true;
                if (seen[0] == window_end[0] && seen[1] == window_end[1]) {
                    if (cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess ||
                        cudaEventQuery(*(events[1][event_ids[1] - 1])) != cudaSuccess)
                    {
                        begin_sche = false;
                    }
                }
                if (begin_sche) {
                    for (int j = 0; j < num_clients; ++j) {
                        if (frecords[j] != NULL && finish_profile[j]) {
                            if (frecords[j]->type != MALLOC_RECORD &&
                                frecords[j]->type != MEMCPY_RECORD &&
                                frecords[j]->type != MEMSET_RECORD &&
                                frecords[j]->type != FREE_RECORD &&
                                num_client_cur_iters[j] > 9)
                            {
                                scheduled_in_tick[j] = true;

                                schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                                events[j][event_ids[j]], seen, event_ids, j);
                                pop_from_queue(client_buffers[j], client_mutexes[j], j);
                            }
                        }
                    }
                }
            }
        }

        // (5.8) Record ONLY MAKESPAN once BOTH finished (in microseconds)
        if (!printed_pair && finished_c[0] && finished_c[1] && !store_profile) {

            auto end0 = finish_ts[0];
            auto end1 = finish_ts[1];
            auto end_max = (end0 > end1) ? end0 : end1;

            long long makespan_ns_ll =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_max - start_profile).count();
            double makespan_us = (double)makespan_ns_ll / 1000.0;

            PairRecord rec;
            rec.iteration    = num_client_cur_iters[0]; // both equal here
            rec.kid0         = window_start[0];
            rec.kid1         = window_start[1];
            rec.tpc0         = schedule_client[0];
            rec.tpc1         = schedule_client[1];
            rec.makespan_us  = makespan_us;
            rec.mask_mode    = is_nomask_row(rec.tpc0, rec.tpc1) ? std::string("nomask") : std::string("masked");
            rec.model0       = model0;
            rec.model1       = model1;

            std::string kname0 = kernel_name_safe(0, rec.kid0);
            std::string kname1 = kernel_name_safe(1, rec.kid1);

            // Print REC with kernel names + microseconds
            printf("[REC] iter=%d kid0=%d name0=%s  kid1=%d name1=%s  "
                   "tpc0=%d tpc1=%d mode=%s makespan_us=%.3f  model0=%s model1=%s\n",
                   rec.iteration,
                   rec.kid0, kname0.c_str(),
                   rec.kid1, kname1.c_str(),
                   rec.tpc0,
                   rec.tpc1,
                   rec.mask_mode.c_str(),
                   rec.makespan_us,
                   rec.model0.c_str(),
                   rec.model1.c_str());

            // ---- Append directly to CSV (microseconds) ----
            try {
                bool need_header = !fs::exists(out_path);

                std::ofstream csv(out_path, std::ios::app);
                if (!csv.is_open()) {
                    std::cerr << "[ERROR] Could not open " << out_path
                              << " for writing." << std::endl;
                } else {
                    if (need_header) {
                        csv << "iteration,model0,model1,kernel0_id,kernel1_id,"
                               "tpc0,tpc1,mask_mode,makespan_us\n";
                    }
                    // keep numeric stable formatting
                    csv.setf(std::ios::fixed);
                    csv << std::setprecision(3);

                    csv << rec.iteration   << ","
                        << rec.model0      << ","
                        << rec.model1      << ","
                        << rec.kid0        << ","
                        << rec.kid1        << ","
                        << rec.tpc0        << ","
                        << rec.tpc1        << ","
                        << rec.mask_mode   << ","
                        << rec.makespan_us << "\n";
                }
            } catch (const std::exception &e) {
                std::cerr << "[ERROR] Exception while writing CSV " << out_path
                          << ": " << e.what() << std::endl;
            }

            store_profile = true;
        }

        // (5.9) When both finished and stored, mark window finished and advance plan
        if (!printed_pair && finished_c[0] && finished_c[1] && store_profile) {
            finish_profile[0] = true;
            finish_profile[1] = true;
            printed_pair      = true;
            store_profile     = false;

            // Advance to next pair (based on iteration index after warmup)
            if (num_client_cur_iters[0] == num_client_cur_iters[1] &&
                num_client_cur_iters[0] > 9 &&
                (num_client_cur_iters[0] - 10) < (int)profile_plan.size())
            {
                auto &next_pair = profile_plan[num_client_cur_iters[0] - 10];
                window_start[0] = next_pair.first.id;
                window_start[1] = next_pair.second.id;
                window_end[0]   = window_start[0] + 1;
                window_end[1]   = window_start[1] + 1;
                schedule_client[0] = next_pair.first.tpc;  // may be -1
                schedule_client[1] = next_pair.second.tpc; // may be -1
            }
        }

        // (5.10) Iteration completion / draining per client
        int finished = 0;
        for (int i = 0; i < num_clients; i++) {
            if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
                (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
                (stop_ack[i] == true))
            {
                finished += 1;
            } else if (seen[i] == num_client_kernels[i]) {
                if (!locked[i]) {
                    pthread_mutex_lock(client_mutexes[i]);
                    locked[i] = true;
                    DEBUG_PRINT("LOCK CLIENT %d\n", i);
                }
                bool ready = true;
                if (seq) {
                    if (event_ids[0] >= 1) {
                        if (cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess)
                            ready = false;
                    }
                } else {
                    if (event_ids[i] >= 1) {
                        if (!warmup) {
                            if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess ||
                                finish_profile[i] == false)
                            {
                                ready = false;
                            }
                        } else {
                            if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess) {
                                ready = false;
                            }
                        }
                    }
                }
                if (ready) {
                    unsetmask(i);

                    finish_profile[i] = false;
                    seen[i]           = 0;
                    if (seq) event_ids[0] = 0;
                    event_ids[i]      = 0;
                    streams[i]        = -1;
                    fidx[i]           = 0;
                    request_status[i][num_client_cur_iters[i]] = true;

                    pthread_mutex_unlock(client_mutexes[i]);
                    num_client_cur_iters[i] += 1;
                    locked[i] = false;
                    client_progress[i] = 0;

                    auto end = std::chrono::high_resolution_clock::now();
                    float ms =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count()
                        / 1000.0f;
                    client_durations[i].push_back(ms);
                }
            }
        }

        if (finished == num_clients) break;
    }

    if (!warmup) {
        auto end_total = std::chrono::high_resolution_clock::now();
        long long total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
        printf("Total loop took %lld microseconds\n", total_us);
    }

    return NULL;
}





void* Scheduler::busy_wait_profile(int num_clients,
                                   int iter,
                                   bool warmup,
                                   int warmup_iters,
                                   int algo,      // now a parameter
                                   bool seq,
                                   int depth,
                                   int hp_limit,
                                   int update_start)
{
    switch (algo) {
        case 0: // ORION
            return busy_wait_orion(num_clients, iter, warmup, warmup_iters,
                                   seq, depth, hp_limit, update_start);

        case 1: // REEF
            return busy_wait_reef(num_clients, iter, warmup, warmup_iters,
                                   seq, depth, hp_limit, update_start);

        case 2: // MULTIPLE STREAM
            return busy_wait_ms(num_clients, iter, warmup, warmup_iters,
                                seq, depth, hp_limit, update_start);

        case 3: // KRISP
            return busy_wait_krisp(num_clients, iter, warmup, warmup_iters,
                                   seq, depth, hp_limit, update_start);

        case 4: // RESERVE (your algorithm) – if you have it
            return busy_wait_ficks(num_clients, iter, warmup, warmup_iters,
                                     seq, depth, hp_limit, update_start);
		case 5: // RESERVE (your algorithm) – if you have it
            return busy_wait_profile_co(num_clients, iter, warmup, warmup_iters,
                                     seq, depth, hp_limit, update_start);

        default:
            printf("Unsupported algo\n");
            abort();
    }
}


extern "C" {

	Scheduler* sched_init() {

		// Scheduler* sched = new Scheduler();
		Scheduler* sched = new Scheduler(2);
		return sched;
	}


	void populate_kernel_info(const char* kernel_info_file,
                          std::vector<op_info>& ops,
                          int algo_id)
    {
        printf("KERNEL_INFO_FILE IS %s\n", kernel_info_file);

        std::ifstream infile(kernel_info_file);
        assert(infile.is_open());

        std::string line;

        // ignore header
        if (!std::getline(infile, line)) {
            infile.close();
            return;
        }

        while (std::getline(infile, line)) {
            if (line.empty()) continue;

            std::vector<std::string> v;
            std::stringstream sline(line);
            while (sline.good()) {
                std::string substr;
                std::getline(sline, substr, ',');
                v.push_back(substr);
            }

            op_info info;  // default constructed

            if (algo_id == ALGO_FICKS) {
                // ===================== FICKS CSV FORMAT =====================
                // 0: id
                // 1: name
                // 2: cluster
                // 3: co-run opt
                // 4: excl opt
                // 5..: excl_us_by_n
                if (v.size() < 5) {
                    fprintf(stderr, "[populate_kernel_info] FICKS line too short: %zu\n",
                            v.size());
                    continue;
                }

                info.id                = std::stoi(v[0]);
                info.name              = v[1];
                info.cluster           = std::stoi(v[2]);
                info.opt_tpc_colocated = std::stoi(v[3]);
                info.opt_tpc_exclusive = std::stoi(v[4]);
                info.is_short = std::stoi(v[5]);
                info.sm_used           = 0;
                info.tpc_used          = 0;

                info.excl_us_by_n.clear();
                info.excl_us_by_n.push_back(
                    std::numeric_limits<float>::quiet_NaN());  // index 0 unused

                for (size_t i = 6; i < v.size(); ++i) {
                    info.excl_us_by_n.push_back(std::stof(v[i]));
                }

            } else {
                // ===================== LEGACY CSV FORMAT ====================
                // 0: name
                // 1: profile
                // 2: mem
                // 3: sm_used
                // 4: duration
                // 5: grid
                // 6: block
                // 7: is on list
                // 8: knee_tpc
                if (v.size() < 8) {
                    fprintf(stderr, "[populate_kernel_info] Legacy line too short: %zu\n",
                            v.size());
                    continue;
                }

                info.name     = v[0];
                info.profile  = std::stoi(v[1]);
                info.mem      = std::stoi(v[2]);
                info.sm_used  = std::stoi(v[3]);
                info.duration = std::stof(v[4]);
                info.grid     = std::stoi(v[5]);
                info.block    = std::stoi(v[6]);
                info.knee_tpc = std::stoi(v[7]);
                // info.is_short = std::stoi(v[8]);
                // info.ratio_ = std::stof(v[9]);
                
            }

            ops.push_back(std::move(info));
        }

        infile.close();
    }

	void setup_change(Scheduler* scheduler, int client_id, char* file, int num_kernels, int algo_id) {

		// needed for backward

		op_info_vector[client_id].clear();
		populate_kernel_info(file, op_info_vector[client_id], algo_id);
		int max_sm_used = 0;
		for (auto info: op_info_vector[client_id])
			max_sm_used = max(max_sm_used, info.sm_used);
		max_sms_clients[client_id] = max_sm_used;
		num_client_kernels[client_id] = num_kernels;

	}


// setup(...)  -- full function
// ===== Add near top of scheduler_eval.cpp (once) =====
#include "critical_model.h"
#include <deque>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

static CriticalityModel g_crit_model;
static std::vector<float> g_crit_thr_per_client;
static bool g_use_predicted_critical_label = true; // true: rewrite op.is_short using prediction

static std::string to_model_name_safe(const char* s) {
    if (!s) return "unknown";
    std::string out(s);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = (char)std::tolower((unsigned char)out[i]);
    }
    // remove suffixes like _8_fwd if they appear in model string
    const std::string fwd = "_fwd";
    size_t p = out.find(fwd);
    if (p != std::string::npos) out = out.substr(0, p);
    return out.empty() ? "unknown" : out;
}

static inline bool is_finite_pos(float x) {
    return std::isfinite(x) && x > 0.0f;
}

static inline float pct(long long a, long long b) {
    if (b <= 0) return 0.0f;
    return 100.0f * (float)a / (float)b;
}


// ===== Entire setup function =====
void setup(
    Scheduler* scheduler,
    int num_clients,
    int* tids,
    char** models,
    char** files,
    int* num_kernels,
    int* num_iters,
    bool* train,
    int   algo_id,   // 0=orion, 1=reef, 2=multistream, 3=krisp, 4=ficks
    float td1,
    float td2
) {
    g_TD1 = td1;
    g_TD2 = td2;

    model_names.clear();
    for (int i = 0; i < num_clients; i++) {
        model_names.emplace_back(models[i]);  // char* -> std::string
    }

    for (const auto& name : model_names) {
        std::cout << "Model: " << name << std::endl;
    }

    const char* SPLIT_MULT_CSV =
        "/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/ficks/profiling/A4000_clusterpair_makespan_mult_lut.csv";
    load_split_mult_once(SPLIT_MULT_CSV);

    struct passwd* pw = getpwuid(getuid());
    (void)pw;
    const char* lib_path = "/home/zixi/orion_bu/src/cuda_capture/libinttemp.so";

    klib = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!klib) {
        fprintf(stderr, "Error: %s\n", dlerror());
        return;
    }

#ifdef SYS_gettid
    pid_t mytid = syscall(SYS_gettid);
#else
#error "SYS_gettid unavailable on this system"
#endif

    // 1) thread structures
    pid_t** thread_ids_all = (pid_t**)dlsym(klib, "thread_ids");
    *thread_ids_all = (pid_t*)malloc((2 * num_clients + 1) * sizeof(pid_t)); // 2*N threads + scheduler

    for (int i = 0; i < num_clients; i++) (*thread_ids_all)[i] = tids[i];
    (*thread_ids_all)[num_clients] = mytid;
    for (int i = num_clients + 1; i < 2 * num_clients + 1; i++) (*thread_ids_all)[i] = 0;

    int** num_total_clients = (int**)dlsym(klib, "num_total_clients");
    *num_total_clients = (int*)malloc(sizeof(int));
    **num_total_clients = num_clients;

    num_cur_clients.resize(num_clients);
    is_executing.resize(num_clients);
    for (int i = 0; i < num_clients; ++i) {
        num_cur_clients[i] = i;
        is_executing[i] = false;
    }
    client_finished = new bool[num_clients](); // false init

    for (int i = 0; i <= num_clients; i++) {
        DEBUG_PRINT("Scheduler setup the thread id at %d to be %d\n", i, (*thread_ids_all)[i]);
    }

    // 2) metadata structures
    op_info_vector.clear();
    client_durations.clear();
    max_sms_clients.clear();
    is_train.clear();
    client_progress.clear();
    func_progress.clear();

    for (int i = 0; i < num_clients; i++) {
        op_info_vector.push_back({});
        client_durations.push_back({});

        populate_kernel_info(files[i], op_info_vector[i], algo_id);

        int max_sm_used = 0;
        for (auto info : op_info_vector[i]) {
            max_sm_used = max(max_sm_used, info.sm_used);
        }
        max_sms_clients.push_back(max_sm_used);

        printf("----------- SIZE: %ld\n", (long)op_info_vector[i].size());

        // keep original meaning of train[] (do not use it to gate predictor training)
        is_train.push_back(train[i]);
        client_progress.push_back(0);
        func_progress.push_back(-1);
    }

    // Save GT labels before any relabel
    std::vector<std::vector<int> > gt_critical(num_clients);
    std::vector<std::vector<int> > gt_cluster(num_clients);
    for (int cid = 0; cid < num_clients; ++cid) {
        const int K = (int)op_info_vector[cid].size();
        gt_critical[cid].resize(K, 0);
        gt_cluster[cid].resize(K, -1);
        for (int k = 0; k < K; ++k) {
            gt_critical[cid][k] = (op_info_vector[cid][k].is_short == 0) ? 1 : 0; // critical=long
            gt_cluster[cid][k] = op_info_vector[cid][k].cluster;
        }
    }

    // ------------------------------------------------------------------
    // Predictor training + eval + relabel (no CSV needed)
    // Inference features used by model:
    //   model name, exact kernel name, sequence bin, past-N predicted ratio bin
    // ------------------------------------------------------------------
    {
        CriticalityModel::Config cfg;
        cfg.alpha = 1.8f;
        cfg.seq_bins = 256;
        cfg.history_n = 20;
        cfg.history_bins = 6;

        cfg.tune_thr_lo = 0.05f;
        cfg.tune_thr_hi = 0.95f;
        cfg.tune_thr_step = 0.01f;
        cfg.mape_floor_us = 10.0f;

        g_crit_model.clear();
        g_crit_model.set_config(cfg);

        // 1) Train once from GT (teacher forcing history with GT labels)
        for (int cid = 0; cid < num_clients; ++cid) {
            const std::string model = to_model_name_safe(models[cid]);
            const int K = (int)op_info_vector[cid].size();

            std::deque<int> hist;
            int hist_pos = 0;

            for (int k = 0; k < K; ++k) {
                const op_info& op = op_info_vector[cid][k];
                const int y = gt_critical[cid][k];      // GT critical
                const int c = gt_cluster[cid][k];       // GT cluster

                const float ratio = hist.empty() ? 0.0f : (float)hist_pos / (float)hist.size();
                const int hbin = g_crit_model.hist_bin_from_ratio(ratio);

                g_crit_model.observe_one(
                    model,
                    op.name,         // exact kernel name
                    k, K,
                    hbin,
                    y,
                    c,
                    op.excl_us_by_n
                );

                hist.push_back(y);
                hist_pos += y;
                if ((int)hist.size() > cfg.history_n) {
                    hist_pos -= hist.front();
                    hist.pop_front();
                }
            }
        }

        struct CritRes { long long correct = 0, total = 0; };
        struct CluRes  { long long correct = 0, total = 0, unknown = 0; };
        struct LatRes  { long long points = 0; double abs_sum = 0.0, ape_sum = 0.0, sape_sum = 0.0; };

        auto eval_client = [&](int cid, float thr, bool eval_cluster_lat) {
            CritRes cr;
            CluRes  zr;
            LatRes  lr;

            const std::string model = to_model_name_safe(models[cid]);
            const int K = (int)op_info_vector[cid].size();

            std::deque<int> hist;
            int hist_pos = 0;

            for (int k = 0; k < K; ++k) {
                const op_info& op = op_info_vector[cid][k];

                const float ratio = hist.empty() ? 0.0f : (float)hist_pos / (float)hist.size();
                const int hbin = g_crit_model.hist_bin_from_ratio(ratio);

                const float p = g_crit_model.predict_critical_prob(model, op.name, k, K, hbin);
                const int yhat = (p >= thr) ? 1 : 0;
                const int ygt  = gt_critical[cid][k];

                cr.total += 1;
                if (yhat == ygt) cr.correct += 1;

                if (eval_cluster_lat) {
                    // cluster
                    const int c_hat = g_crit_model.predict_cluster(model, op.name, k, K, hbin);
                    const int c_gt  = gt_cluster[cid][k];
                    zr.total += 1;
                    if (c_hat < 0) zr.unknown += 1;
                    if (c_hat == c_gt) zr.correct += 1;

                    // latency curve at each tpc point
                    for (int t = 1; t <= (int)op.excl_us_by_n.size(); ++t) {
                        const float gt = op.excl_us_by_n[t - 1];
                        if (!is_finite_pos(gt)) continue;

                        const float pr = g_crit_model.predict_latency_at_tpc(model, op.name, k, K, hbin, t);
                        if (!is_finite_pos(pr)) continue;

                        const double gtd = (double)gt;
                        const double prd = (double)pr;
                        const double ae = std::fabs(prd - gtd);
                        const double denom = std::max(gtd, (double)cfg.mape_floor_us);
                        const double ape = ae / denom;
                        const double sape = ae / (std::fabs(prd) + std::fabs(gtd) + 1e-9);

                        lr.points += 1;
                        lr.abs_sum += ae;
                        lr.ape_sum += ape;
                        lr.sape_sum += sape;
                    }
                }

                // autoregressive history update with predicted label
                hist.push_back(yhat);
                hist_pos += yhat;
                if ((int)hist.size() > cfg.history_n) {
                    hist_pos -= hist.front();
                    hist.pop_front();
                }
            }

            return std::make_tuple(cr, zr, lr);
        };

        // 2) Tune threshold per client on accuracy
        g_crit_thr_per_client.assign(num_clients, 0.5f);
        std::vector<CritRes> all_default(num_clients), all_tuned(num_clients);
        std::vector<CluRes> all_clu(num_clients);
        std::vector<LatRes> all_lat(num_clients);

        for (int cid = 0; cid < num_clients; ++cid) {
            float best_thr = 0.5f;
            long long best_correct = -1;
            long long best_total = 1;

            for (float thr = cfg.tune_thr_lo; thr <= cfg.tune_thr_hi + 1e-6f; thr += cfg.tune_thr_step) {
                auto tup = eval_client(cid, thr, false);
                CritRes cr = std::get<0>(tup);

                if (cr.correct > best_correct) {
                    best_correct = cr.correct;
                    best_total = cr.total;
                    best_thr = thr;
                }
            }

            g_crit_thr_per_client[cid] = best_thr;

            auto d = eval_client(cid, 0.5f, true);
            auto t = eval_client(cid, best_thr, true);

            all_default[cid] = std::get<0>(d);
            all_tuned[cid]   = std::get<0>(t);
            all_clu[cid]     = std::get<1>(t);
            all_lat[cid]     = std::get<2>(t);

            const std::string model = to_model_name_safe(models[cid]);

            printf("[CRIT-EVAL] client=%d model=%s (default) | thr=0.500 acc=%.2f%% (%lld/%lld)\n",
                   cid, model.c_str(),
                   pct(all_default[cid].correct, all_default[cid].total),
                   all_default[cid].correct, all_default[cid].total);

            printf("[CRIT-EVAL] client=%d model=%s (tuned) | thr=%.3f acc=%.2f%% (%lld/%lld)\n",
                   cid, model.c_str(), best_thr,
                   pct(all_tuned[cid].correct, all_tuned[cid].total),
                   all_tuned[cid].correct, all_tuned[cid].total);

            printf("[CLUSTER-EVAL] client=%d model=%s | acc=%.2f%% (%lld/%lld) unknown_pred=%lld\n",
                   cid, model.c_str(),
                   pct(all_clu[cid].correct, all_clu[cid].total),
                   all_clu[cid].correct, all_clu[cid].total, all_clu[cid].unknown);

            if (all_lat[cid].points > 0) {
                const double mae   = all_lat[cid].abs_sum / (double)all_lat[cid].points;
                const double mape  = 100.0 * all_lat[cid].ape_sum / (double)all_lat[cid].points;
                const double smape = 200.0 * all_lat[cid].sape_sum / (double)all_lat[cid].points;
                printf("[LAT-EVAL] client=%d model=%s | points=%lld MAE=%.4fus MAPE=%.2f%% sMAPE=%.2f%%\n",
                       cid, model.c_str(), all_lat[cid].points, mae, mape, smape);
            } else {
                printf("[LAT-EVAL] client=%d model=%s | points=0\n", cid, model.c_str());
            }
        }

        // 3) overall micro
        CritRes odef, otun;
        CluRes  oclu;
        LatRes  olat;

        for (int cid = 0; cid < num_clients; ++cid) {
            odef.correct += all_default[cid].correct; odef.total += all_default[cid].total;
            otun.correct += all_tuned[cid].correct;   otun.total += all_tuned[cid].total;

            oclu.correct += all_clu[cid].correct;     oclu.total += all_clu[cid].total; oclu.unknown += all_clu[cid].unknown;

            olat.points += all_lat[cid].points;
            olat.abs_sum += all_lat[cid].abs_sum;
            olat.ape_sum += all_lat[cid].ape_sum;
            olat.sape_sum += all_lat[cid].sape_sum;
        }

        printf("[CRIT-EVAL] OVERALL_MICRO model=all (default) | thr=0.500 acc=%.2f%% (%lld/%lld)\n",
               pct(odef.correct, odef.total), odef.correct, odef.total);

        printf("[CRIT-EVAL] OVERALL_MICRO (tuned-per-client) | acc=%.2f%% (%lld/%lld)\n",
               pct(otun.correct, otun.total), otun.correct, otun.total);

        printf("[CLUSTER-EVAL] OVERALL_MICRO model=all | acc=%.2f%% (%lld/%lld) unknown_pred=%lld\n",
               pct(oclu.correct, oclu.total), oclu.correct, oclu.total, oclu.unknown);

        if (olat.points > 0) {
            const double mae   = olat.abs_sum / (double)olat.points;
            const double mape  = 100.0 * olat.ape_sum / (double)olat.points;
            const double smape = 200.0 * olat.sape_sum / (double)olat.points;
            printf("[LAT-EVAL] OVERALL_MICRO model=all | points=%lld MAE=%.4fus MAPE=%.2f%% sMAPE=%.2f%%\n",
                   olat.points, mae, mape, smape);
        } else {
            printf("[LAT-EVAL] OVERALL_MICRO model=all | points=0\n");
        }

        // 4) Write predicted critical label back to op_info (for runtime scheduler)
        for (int cid = 0; cid < num_clients; ++cid) {
            const std::string model = to_model_name_safe(models[cid]);
            const int K = (int)op_info_vector[cid].size();
            const float thr = (cid < (int)g_crit_thr_per_client.size()) ? g_crit_thr_per_client[cid] : 0.5f;

            std::deque<int> hist;
            int hist_pos = 0;

            for (int k = 0; k < K; ++k) {
                op_info& op = op_info_vector[cid][k];

                const float ratio = hist.empty() ? 0.0f : (float)hist_pos / (float)hist.size();
                const int hbin = g_crit_model.hist_bin_from_ratio(ratio);

                const float p = g_crit_model.predict_critical_prob(model, op.name, k, K, hbin);
                const int yhat = (p >= thr) ? 1 : 0; // 1 means critical/long

                op.p_critical = p;
                op.pred_is_critical = yhat;

                if (g_use_predicted_critical_label) {
                    op.is_short = yhat ? 0 : 1; // relabel runtime long/short
                }

                hist.push_back(yhat);
                hist_pos += yhat;
                if ((int)hist.size() > cfg.history_n) {
                    hist_pos -= hist.front();
                    hist.pop_front();
                }
            }
        }
    }

    // 3) indexes
    int** fidx_ptr = (int**)dlsym(klib, "func_indexes");
    *fidx_ptr = (int*)calloc(num_clients, sizeof(int));
    fidx = *fidx_ptr;

    num_client_kernels = num_kernels;
    num_client_max_iters = num_iters;

    num_client_cur_iters = (int*)calloc(num_clients, sizeof(int));
    locked = (bool*)calloc(num_clients, sizeof(bool));

    // measurements
    client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)
        calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
    total_client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)
        calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));

    client_starts_set = (bool**)malloc(num_clients * sizeof(bool*));
    for (int i = 0; i < num_clients; i++) {
        client_starts_set[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
    }

    // 4) communication queues + locks
    queue<func_record>*** buffers_ptr = (queue<func_record>***)dlsym(klib, "kqueues");
    *buffers_ptr = (queue<func_record>**)malloc(num_clients * sizeof(queue<func_record>*));
    queue<func_record>** buffers = *buffers_ptr;
    for (int i = 0; i < num_clients; i++) {
        buffers[i] = new queue<func_record>();
        printf("buffer size is %ld\n", (long)buffers[i]->size());
    }

    pthread_mutex_t*** client_mutexes_ptr = (pthread_mutex_t***)dlsym(klib, "mutexes");
    *client_mutexes_ptr = (pthread_mutex_t**)malloc(num_clients * sizeof(pthread_mutex_t*));
    client_mutexes = *client_mutexes_ptr;
    for (int i = 0; i < num_clients; i++) {
        client_mutexes[i] = new pthread_mutex_t();
    }

    if (algo_id == ALGO_REEF) {
        scheduler->profile_prep(buffers, num_clients, 1);
    } else {
        scheduler->profile_prep(buffers, num_clients, 0);
    }

    // 5) runtime control
    bool*** request_status_ptr = (bool***)dlsym(klib, "client_request_status");
    *request_status_ptr = (bool**)malloc(num_clients * sizeof(bool*));
    request_status = *request_status_ptr;

    bool** stops_ptr = (bool**)dlsym(klib, "client_stop");
    *stops_ptr = (bool*)calloc(num_clients, sizeof(bool));
    stops = *stops_ptr;

    bool** stop_ack_ptr = (bool**)dlsym(klib, "client_stop_ack");
    *stop_ack_ptr = (bool*)calloc(num_clients, sizeof(bool));
    stop_ack = *stop_ack_ptr;

    bool** affinity_set_ptr = (bool**)dlsym(klib, "affinity_set");
    (*affinity_set_ptr) = (bool*)calloc(num_clients + 1, sizeof(bool));

    for (int i = 0; i < num_clients; i++) {
        request_status[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
    }
}







void* schedule(Scheduler* scheduler,
               int num_clients,
               bool profile_mode,
               int iter,
               bool warmup,
               int warmup_iters,
               int algo,        // 0..4
               int depth,       // reef_depth / orion_max_be_duration / etc.
               int hp_limit,
               int update_start)
{
    printf("entered sched func!\n");
    if (profile_mode) {
        bool seq = false;
        scheduler->busy_wait_profile(num_clients,
                                     iter,
                                     warmup,
                                     warmup_iters,
                                     algo,      // pass through
                                     seq,
                                     depth,
                                     hp_limit,
                                     update_start);
    }
    printf("exited sched func!\n");
    return NULL;
}


	void* reset(Scheduler* scheduler, int num_clients) {
		scheduler->profile_reset(num_clients);
		return NULL;
	}
}