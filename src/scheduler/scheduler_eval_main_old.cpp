#include "scheduler.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip> 
#include <unordered_map>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

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

inline double pair_ratio(int ut, int co, int n) {
    if (use_excl) return 1.0;

    if (!g_loaded) {
        std::fprintf(stderr, "[ERR] pair_ratio called before tables are loaded\n");
        std::abort();
    }

    if (n < 1 || n > HW_NUM_TPCS) {
        std::fprintf(stderr, "[ERR] pair_ratio: invalid n=%d (HW_NUM_TPCS=%d)\n",
                     n, HW_NUM_TPCS);
        std::abort();
    }
    if (ut < 0 || co < 0) {
        std::fprintf(stderr, "[ERR] pair_ratio: invalid ut=%d co=%d\n", ut, co);
        std::abort();
    }

    if (ut >= (int)g_pair.size() ||
        co >= (int)g_pair[ut].size() ||
        n  >= (int)g_pair[ut][co].size())
    {
        // missing pair entry is normal -> return 1.0 so caller can fallback to UT
        return 1.0;
    }

    double v = g_pair[ut][co][n];
    if (!std::isfinite(v) || v <= 0.0) {
        // missing value -> return 1.0 so caller uses UT fallback
        return 1.0;
    }
    return v;
}


inline double ut_ratio(int ut, int n) {
    if (use_excl) return 1.0;

    if (!g_loaded) {
        std::fprintf(stderr, "[ERR] ut_ratio called before tables are loaded\n");
        std::abort();
    }

    if (n < 1 || n > HW_NUM_TPCS) {
        std::fprintf(stderr, "[ERR] ut_ratio: invalid n=%d (HW_NUM_TPCS=%d)\n",
                     n, HW_NUM_TPCS);
        std::abort();
    }
    if (ut < 0) {
        std::fprintf(stderr, "[ERR] ut_ratio: invalid ut=%d\n", ut);
        std::abort();
    }

    // Missing UT row -> fallback silently
    if (ut >= (int)g_ut.size()) {
        return 1.0;
    }

    // Missing n in this UT row -> fallback silently
    if (n >= (int)g_ut[ut].size()) {
        return 1.0;
    }

    double v = g_ut[ut][n];
    if (std::isfinite(v) && v > 0.0) return v;

    // -------- nearest-neighbor fallback (silent) --------
    // Search downward first (n-1, n-2, ...)
    for (int k = n - 1; k >= 1; --k) {
        if (k < (int)g_ut[ut].size()) {
            double vv = g_ut[ut][k];
            if (std::isfinite(vv) && vv > 0.0) {
                return vv;
            }
        }
    }

    // Search upward (n+1, n+2, ...)
    for (int k = n + 1; k <= HW_NUM_TPCS; ++k) {
        if (k < (int)g_ut[ut].size()) {
            double vv = g_ut[ut][k];
            if (std::isfinite(vv) && vv > 0.0) {
                return vv;
            }
        }
    }

    // Nothing usable in this UT row -> fallback silently
    return 1.0;
}



// ---- One-shot loader: pass the exact directory with the CSVs ----
inline void load_ratio_tables_once(const char* latency_root_dir)
{
    if (use_excl) {
        g_pair.clear();
        g_ut.clear();
        g_loaded = true;
        std::cout << "[INFO] use_excl=true: skipping ratio CSVs, "
                  << "all pair_ratio/ut_ratio will be 1.0\n";
        return;
    }

    if (g_loaded) return;

    namespace fs = std::filesystem;
    fs::path root(latency_root_dir);
    if (!fs::exists(root) || !fs::is_directory(root)) {
        std::fprintf(stderr, "[ERR] latency root not found: %s\n", latency_root_dir);
        std::abort();
    }

    auto split_csv = [](const std::string& s) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            size_t a = 0, b = tok.size();
            while (a < b && std::isspace((unsigned char)tok[a])) ++a;
            while (b > a && std::isspace((unsigned char)tok[b-1])) --b;
            out.emplace_back(tok.substr(a, b - a));
        }
        return out;
    };

    auto idxOf = [](const std::vector<std::string>& cols, const std::string& name){
        for (size_t i = 0; i < cols.size(); ++i)
            if (cols[i] == name) return (int)i;
        return -1;
    };

    auto normalize_n = [](int n) {
        n = std::max(1, std::min(num_tpcs, n));
        // You profile n=1,2,4,6,... but not 3,5,7,...
        if (n > 1 && (n & 1)) --n;   // 7 -> 6
        return n;
    };

    g_pair.clear();
    g_ut.clear();
    bool any = false;

    // ==========================================================
    // PAIR TABLE
    // ==========================================================
    {
        fs::path csv_path = root / "A4000_ratios_pair.csv";
        std::ifstream f(csv_path.string());
        if (!f) {
            std::fprintf(stderr, "[ERR] missing %s\n", csv_path.string().c_str());
            std::abort();
        }

        std::string line;
        if (!std::getline(f, line)) {
            std::fprintf(stderr, "[ERR] empty %s\n", csv_path.string().c_str());
            std::abort();
        }

        auto hdr = split_csv(line);
        int i_cu = idxOf(hdr, "cluster_UT");
        int i_cc = idxOf(hdr, "cluster_CO");
        int i_n  = idxOf(hdr, "n");
        int i_mu = idxOf(hdr, "mult");
        int i_rg = idxOf(hdr, "ratio_gm");

        // REQUIRED columns check (do not silently load garbage)
        if (i_cu < 0 || i_cc < 0 || i_n < 0 || (i_mu < 0 && i_rg < 0)) {
            std::fprintf(stderr,
                         "[ERR] bad header in %s. Need cluster_UT, cluster_CO, n, and (mult or ratio_gm)\n",
                         csv_path.string().c_str());
            std::abort();
        }

        size_t rows = 0, ok = 0;

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            ++rows;
            auto c = split_csv(line);
            if ((int)c.size() <= std::max({i_cu, i_cc, i_n, i_mu, i_rg})) continue;

            int cu = std::stoi(c[i_cu]);
            int cc = std::stoi(c[i_cc]);
            int n  = normalize_n(std::stoi(c[i_n]));

            double mult = 1.0;
            if (i_mu >= 0 && !c[i_mu].empty()) {
                mult = std::stod(c[i_mu]);
            } else if (i_rg >= 0 && !c[i_rg].empty()) {
                mult = 1.0 + std::stod(c[i_rg]);
            }

            if (cu >= (int)g_pair.size()) g_pair.resize(cu + 1);
            if (cc >= (int)g_pair[cu].size()) g_pair[cu].resize(cc + 1);
            if (n  >= (int)g_pair[cu][cc].size())
                g_pair[cu][cc].resize(n + 1, 0.0);

            if (std::isfinite(mult) && mult > 0.0) {
                g_pair[cu][cc][n] = mult;
                ++ok;
                any = true;
            }
        }

        std::cout << "[INFO] loaded global ratios_pair.csv rows="
                  << rows << " ok=" << ok << "\n";
    }

    // ==========================================================
    // UT TABLE
    // ==========================================================
    {
        fs::path csv_path = root / "A4000_ratios_ut.csv";
        std::ifstream f(csv_path.string());
        if (!f) {
            std::fprintf(stderr, "[ERR] missing %s\n", csv_path.string().c_str());
            std::abort();
        }

        std::string line;
        if (!std::getline(f, line)) {
            std::fprintf(stderr, "[ERR] empty %s\n", csv_path.string().c_str());
            std::abort();
        }

        auto hdr = split_csv(line);
        int i_cu = idxOf(hdr, "cluster_UT");
        int i_n  = idxOf(hdr, "n");
        int i_mu = idxOf(hdr, "mult_ut");
        int i_rg = idxOf(hdr, "ratio_ut_gm");

        if (i_cu < 0 || i_n < 0 || (i_mu < 0 && i_rg < 0)) {
            std::fprintf(stderr,
                         "[ERR] bad header in %s. Need cluster_UT, n, and (mult_ut or ratio_ut_gm)\n",
                         csv_path.string().c_str());
            std::abort();
        }

        size_t rows = 0, ok = 0;

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            ++rows;
            auto c = split_csv(line);
            if ((int)c.size() <= std::max({i_cu, i_n, i_mu, i_rg})) continue;

            int cu = std::stoi(c[i_cu]);
            int n  = normalize_n(std::stoi(c[i_n]));

            double mult = 1.0;
            if (i_mu >= 0 && !c[i_mu].empty()) {
                mult = std::stod(c[i_mu]);
            } else if (i_rg >= 0 && !c[i_rg].empty()) {
                mult = 1.0 + std::stod(c[i_rg]);
            }

            if (cu >= (int)g_ut.size()) g_ut.resize(cu + 1);
            if (n  >= (int)g_ut[cu].size())
                g_ut[cu].resize(n + 1, 0.0);

            if (std::isfinite(mult) && mult > 0.0) {
                g_ut[cu][n] = mult;
                ++ok;
                any = true;
            }
        }

        std::cout << "[INFO] loaded global ratios_ut.csv rows="
                  << rows << " ok=" << ok << "\n";
    }

    if (!any) {
        std::fprintf(stderr, "[ERR] no ratio rows loaded under %s\n",
                     root.string().c_str());
        std::abort();
    }

    g_loaded = true;
    std::cout << "[OK] Loaded GLOBAL ratio tables from " << root << "\n";

    // (your debug preview code stays the same)
}






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
void* Scheduler::busy_wait_orion(int num_clients,
                                 int iter,
                                 bool warmup,
                                 int warmup_iters,
                                 bool seq,
                                 int depth,
                                 int hp_limit,
                                 int update_start)
{
    printf("Entered busy_wait_orion! Num clients = %d\n", num_clients);

    auto start_total = std::chrono::high_resolution_clock::now();
    std::vector<bool> total_client_set(num_clients, false);

    // ------------------------------------------------------------
    // Round-robin HP pointer (persists across calls)
    // Start it from priority_client if valid, otherwise 0.
    // ------------------------------------------------------------
    static int rr_next_hp = -1;
    if (rr_next_hp < 0) {
        if (priority_client >= 0 && priority_client < num_clients) rr_next_hp = priority_client;
        else rr_next_hp = 0;
    }

    while (true) {
        std::vector<func_record*> frecords(num_clients, nullptr);

        // 1) Pull one record per client, set start timestamps
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

        // 2) Check events & build canSchedule[]
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

        // 3) ORION scheduling step
        // ----------------------------------------------------
        // 3.1 Admin ops: always schedule, no mask
        for (int i = 0; i < num_clients; ++i) {
            func_record* rec = frecords[i];
            if (!rec) continue;
            if (!is_admin_record(*rec)) continue;

            schedule_kernel(*rec, sched_streams[i], i,
                            events[i][event_ids[i]], seen, event_ids, i);
            pop_from_queue(client_buffers[i], client_mutexes[i], i);
            frecords[i] = nullptr;  // consumed
        }

        // ----------------------------------------------------
        // 3.2 HP client: STRICT ROUND ROBIN among eligible clients
        // ----------------------------------------------------
        int hp_client = -1;

        // pick first eligible client starting from rr_next_hp
        for (int k = 0; k < num_clients; ++k) {
            int cand = (rr_next_hp + k) % num_clients;
            func_record* rec = frecords[cand];
            if (!rec) continue;
            if (!canSchedule[cand]) continue;
            if (is_admin_record(*rec)) continue;  // admin already handled
            hp_client = cand;
            break;
        }

        long long hp_endT = 0;      // approximate HP completion window
        int hp_tpcs_used  = 0;      // track how many TPCs HP uses
        long long hp_op_duration = 0;

        if (hp_client >= 0) {
            // snapshot HP op info BEFORE scheduling (seen[] might change after schedule_kernel)
            op_info op_hp = op_info_vector[hp_client][seen[hp_client]];

            int tpc_usage = op_hp.sm_used / 2;
            if (tpc_usage < 1) tpc_usage = 1;
            if (tpc_usage > num_tpcs) tpc_usage = num_tpcs;

            hp_tpcs_used = tpc_usage;
            hp_op_duration = (long long)op_hp.duration;

            if (tpc_usage > 0) {
                setmask(tpc_usage, hp_client);

                schedule_kernel(*frecords[hp_client],
                                sched_streams[hp_client],
                                hp_client,
                                events[hp_client][event_ids[hp_client]],
                                seen, event_ids, hp_client);

                pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
                frecords[hp_client] = nullptr;

                // approximate HP finish time using threshold td1
                double td1 = 0.8;  // tune if you like
                auto now = std::chrono::high_resolution_clock::now();
                long long now_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now.time_since_epoch()).count();
                hp_endT = now_ns + (long long)(td1 * (double)hp_op_duration);

                // advance RR pointer ONLY when HP actually launches
                rr_next_hp = (hp_client + 1) % num_clients;
            } else {
                // HP candidate invalid -> treat as no HP this round
                hp_client = -1;
            }
        }

        // 3.3 BE clients: co-run with HP if there are TPCs left
        if (hp_endT > 0) {
            int remaining_tpcs = num_tpcs - hp_tpcs_used;
            if (remaining_tpcs < 0) remaining_tpcs = 0;

            for (int i = 0; i < num_clients; ++i) {
                if (i == hp_client) continue;
                func_record* rec = frecords[i];
                if (!rec || !canSchedule[i]) continue;
                if (is_admin_record(*rec)) continue;

                op_info op_be = op_info_vector[i][seen[i]];

                // simple td2 test: if BE is not too long vs HP, co-run
                double td2 = 2.0;  // tune if you like
                if (hp_op_duration > 0 &&
                    (double)op_be.duration > td2 * (double)hp_op_duration)
                {
                    continue;
                }

                int tpc_usage = op_be.sm_used / 2;
                if (tpc_usage < 1) tpc_usage = 1;
                if (tpc_usage > remaining_tpcs) tpc_usage = remaining_tpcs;
                if (tpc_usage <= 0) continue;

                setmask(tpc_usage, i);

                schedule_kernel(*rec,
                                sched_streams[i],
                                i,
                                events[i][event_ids[i]],
                                seen, event_ids, i);

                pop_from_queue(client_buffers[i], client_mutexes[i], i);
                frecords[i] = nullptr;

                remaining_tpcs -= tpc_usage;
                if (remaining_tpcs <= 0) break;
            }
        } else {
            // HP not launched this round: let BE run if they can
            for (int i = 0; i < num_clients; ++i) {
                func_record* rec = frecords[i];
                if (!rec || !canSchedule[i]) continue;
                if (is_admin_record(*rec)) continue;

                op_info op_be = op_info_vector[i][seen[i]];
                int tpc_usage = op_be.sm_used / 2;
                if (tpc_usage < 1) tpc_usage = 1;
                if (tpc_usage > num_tpcs) tpc_usage = num_tpcs;
                if (tpc_usage <= 0) continue;

                setmask(tpc_usage, i);

                schedule_kernel(*rec,
                                sched_streams[i],
                                i,
                                events[i][event_ids[i]],
                                seen, event_ids, i);

                pop_from_queue(client_buffers[i], client_mutexes[i], i);
                frecords[i] = nullptr;
            }
        }

        // 4) iteration completion logic
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
        printf("Total ORION loop took %lld ns\n", duration_ns);
    }

    return NULL;
}





// Multiple Streams
void* Scheduler::busy_wait_ms(int num_clients, int iter, bool warmup, int warmup_iters,
                                    bool seq, int depth, int hp_limit, int update_start)
{


    printf("Entered busy_wait_profile (MS, no profiling)! Num clients is %d\n", num_clients);

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
        bool canSchedule[num_clients];
        for (int i = 0; i < num_clients; ++i) {
            canSchedule[i] = true;
            if (event_ids[i] >= 1) {
                if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                    // Previous kernel finished: release TPC mask
                    unsetmask(i);
                } else {
                    canSchedule[i] = false;
                }
            }
        }

        // ----------------------------
        // Schedule ready work
        // ----------------------------
        for (int j = 0; j < num_clients; ++j) {
            if (frecords[j] == NULL) continue;

            func_record* rec = frecords[j];
            // Compute kernels: only if previous one has finished
            if (!canSchedule[j]) continue;
            
            // Non-compute records: schedule directly, no mask
            if (rec->type == MALLOC_RECORD ||
                rec->type == MEMCPY_RECORD ||
                rec->type == MEMSET_RECORD ||
                rec->type == FREE_RECORD)
            {
                schedule_kernel(*rec, sched_streams[j], j,
                                events[j][event_ids[j]],
                                seen, event_ids, j);
                pop_from_queue(client_buffers[j], client_mutexes[j], j);
                continue;
            }
            // Multiple-stream baseline: use a fixed TPC share (e.g., 32)
            // You can change 32 if you want a different partition.
            // setmask(24/num_clients, j);

            schedule_kernel(*rec, sched_streams[j], j,
                            events[j][event_ids[j]],
                            seen, event_ids, j);
            pop_from_queue(client_buffers[j], client_mutexes[j], j);
        }

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

            if (is_executing[i] == true) {
                continue;
            }

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
        bool canSchedule[num_clients];
        for (int i = 0; i < num_clients; ++i) {
            canSchedule[i] = true;
            if (event_ids[i] >= 1) {
                if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                    unsetmask(i);
                } else {
                    canSchedule[i] = false;
                }
            }
        }

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
            if (frecords[hp_client] != NULL && canSchedule[hp_client]) {

                if (frecords[hp_client]->type != MALLOC_RECORD &&
                    frecords[hp_client]->type != MEMCPY_RECORD &&
                    frecords[hp_client]->type != MEMSET_RECORD &&
                    frecords[hp_client]->type != FREE_RECORD) {

                    int hp_idx = seen[hp_client];
                    op_info op_info_1 = op_info_vector[hp_client][hp_idx];

                    int tpc_usage = op_info_1.sm_used / 2;
                    tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;

                    if (num_tpcs >= tpc_usage) {
                        setmask(tpc_usage, hp_client);

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

                        if (!canSchedule[i]) continue;

                        op_info op_info_0 = op_info_vector[i][seen[i]];
                        int tpc_usage = op_info_0.sm_used / 2;
                        tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;

                        if (num_tpcs > 0) {
                            int assigned = std::min(num_tpcs, tpc_usage);
                            setmask(assigned, i);

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
                        if (frecords[i] != NULL && canSchedule[i] && i != hp_client) {

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
                                setmask(assigned, i);

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
                    unsetmask(i);
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
                    setmask(assigned_tpc, j);

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

//         vector<func_record*> frecords(num_clients, NULL);

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

//         // warmup gate (keep your original behavior)
//         bool can_schedule = true;
//         for (int j = 0; j < num_clients; ++j) {
//             if (num_client_cur_iters[j] < 10) {
//                 can_schedule = false;
//                 break;
//             }
//         }

//         // Build list of compute-ready clients whose LAST GPU kernel finished
//         std::vector<int> ready_client;
//         ready_client.reserve(num_clients);

//         for (int j = 0; j < num_clients; ++j) {

//             if (frecords[j] == NULL) continue;

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

//             // compute ops: only consider after warmup iterations
//             if (num_client_cur_iters[j] <= 9) continue;

//             // "last finished" check:
//             // - if event_ids[j] == 0 => no previous kernel recorded => ready
//             // - else previous event must be complete
//             bool last_finished = true;
//             if (event_ids[j] >= 1) {
//                 if (cudaEventQuery(*(events[j][event_ids[j] - 1])) != cudaSuccess) {
//                     // last_finished = false;
//                 }
//             }

//             if (last_finished) {
//                 ready_client.push_back(j);
//             }
//         }

//         // Only when BOTH (>=2) clients are ready AND past warmup: schedule compute kernels
//         if (can_schedule && ready_client.size() > 1) {

//             // Now that we KNOW we will schedule, we can safely unset masks for these clients
//             // (optional, but consistent with your intent)
//             for (int client_id : ready_client) {
//                 unsetmask(client_id);
//             }

//             // Schedule all ready compute clients (or only two if you want strict pairing)
//             for (int client_id : ready_client) {
//                 schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id,
//                                 events[client_id][event_ids[client_id]],
//                                 seen, event_ids, client_id);
//                 pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
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
//                     duration /= 1000.0;
//                     client_durations[i].push_back(duration);
//                 }
//             }
//         }

//         if (finished == num_clients) break;
//     }

//     if (!warmup) {
//         auto end_total = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
//         duration /= 1000.0;
//         printf("Total loop took %ld nanoseconds\n", duration);
//     }

//     return NULL;
// }




struct ProfileCheck {
  int   TPC_IND_OPT;     // #TPC_IND_OPT  (exclusive knee)
  int   TPC_COR_OPT;     // #TPC_COR_OPT  (colocation knee)
  float TIME_IND_OPT;    // time at exclusive knee
  float TIME_COR_OPT;    // time at coloc knee (worst-case vs running)
  float TIME_COR_AVAIL;  // time at available TPCs (worst-case vs running)
};

// safe read (expects excl_us_by_n indexed by exact TPC n; element 0 may be unused)
static inline float excl_at_n(const op_info* op, int n) {
  if (!op) return NAN;
  if (n >= 0 && n < (int)op->excl_us_by_n.size()) {
    float v = op->excl_us_by_n[n];
    return (std::isfinite(v) && v > 0.f) ? v : NAN;
  }
  return NAN;
}

static inline int client_active_tpcs(int client_id)
{
    int tpcs = 0;
    for (const auto& rk : g_running_ops) {
        if (rk.client_id == client_id && rk.op && rk.op->is_running) {
            tpcs = std::max(tpcs, rk.op->tpc_used);
        }
    }
    return tpcs;
}


static inline ProfileCheck profile_check(
    const op_info* cur,
    const std::vector<const op_info*>& running,
    int existing_tpcs,
    int tpc_avail_total
)
{
    ProfileCheck out{};
    if (!cur) return out;

    const int n_excl_opt  = std::max(1, cur->opt_tpc_exclusive);
    const int n_coloc_opt = std::max(1, cur->opt_tpc_colocated);

    out.TPC_IND_OPT = n_excl_opt;
    out.TPC_COR_OPT = n_coloc_opt;

    // -------- TIME_IND_OPT at exclusive knee --------
    out.TIME_IND_OPT = excl_at_n(cur, n_excl_opt);

    // -------- TIME_COR_OPT at coloc knee --------
    const double base_ut = ut_ratio(cur->cluster, n_coloc_opt);

    double worst_ratio = base_ut;
    for (const auto* r : running) {
        if (!r) continue;
        const double r_pair = pair_ratio(cur->cluster, r->cluster, n_coloc_opt);
        const double rco    = std::max(r_pair, base_ut);
        if (rco > worst_ratio) worst_ratio = rco;
    }

    const float t_excl_cor = excl_at_n(cur, n_coloc_opt);
    out.TIME_COR_OPT = (std::isfinite(t_excl_cor) ? float(t_excl_cor * worst_ratio) : NAN);

    // ============================================================
    // AVAILABLE PART: use tpc_avail_total directly
    // (NO forced >= existing_tpcs)
    // ============================================================
    int n_avail = std::min(n_coloc_opt, std::max(1, tpc_avail_total));
    if (n_avail < 1) n_avail = 1;

    const double base_ut_av = ut_ratio(cur->cluster, n_avail);

    double worst_ratio_av = base_ut_av;
    for (const auto* r : running) {
        if (!r) continue;
        const double r_pair_av = pair_ratio(cur->cluster, r->cluster, n_avail);
        const double rco_av    = std::max(r_pair_av, base_ut_av);
        if (rco_av > worst_ratio_av) worst_ratio_av = rco_av;
    }

    const float t_excl_av = excl_at_n(cur, n_avail);
    out.TIME_COR_AVAIL = (std::isfinite(t_excl_av) ? float(t_excl_av * worst_ratio_av) : NAN);

    return out;
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
    tpc_mask_t owned = localMask[idx];
    int have = popcount_mask(owned);

    if (have <= needed_tpcs) {
        return;
    }

    int to_release = have - needed_tpcs;

    for (int i = HW_NUM_TPCS - 1; i >= 0 && to_release > 0; --i) {
        if ((owned >> i) & tpc_mask_t(1)) {
            owned &= ~(tpc_mask_t(1) << i);

            // ⚠️ check your mask semantics here
            mask  &= ~(tpc_mask_t(1) << i);

            if (tpc_usage_count[i] > 0) {
                --tpc_usage_count[i];
            }
            ++num_tpcs;  // assuming this counts free TPCs
            --to_release;
        }
    }

    localMask[idx] = owned;
}


static inline void update_and_build_running_ops(
    uint64_t now_ns,
    int exclude_client_id,
    std::vector<const op_info*>& out_running)
{
    // using RunningOp = Scheduler::RunningOp;
    auto& vec = g_running_ops;

    out_running.clear();
    out_running.reserve(vec.size());

    std::vector<RunningOp> new_vec;
    new_vec.reserve(vec.size());

    const int max_clients = (int)op_info_vector.size();
    std::vector<int> needed_tpcs(max_clients, 0);

    auto is_finished = [&](const RunningOp& rk) -> bool {
        op_info* op = rk.op;
        if (!op) return true;
        if (!op->is_running) return true;
        if (op->est_finish_ns <= now_ns) return true;
        return false;
    };

    for (const auto& rk : vec) {

        if (is_finished(rk)) {
            if (rk.op) rk.op->is_running = false;
            continue;
        }

        // still running
        new_vec.push_back(rk);

        // per-client max mask size needed
        if (rk.client_id >= 0 && rk.client_id < max_clients && rk.op) {
            needed_tpcs[rk.client_id] = std::max(needed_tpcs[rk.client_id], rk.op->tpc_used);
        }

        // build out_running excluding exclude_client_id
        if (rk.client_id != exclude_client_id && rk.op) {
            out_running.push_back(rk.op);
        }
    }

    // shrink/unset mask ONCE per client based on remaining running ops
    for (int cid = 0; cid < max_clients; ++cid) {
        int need = needed_tpcs[cid];
        if (need == 0) {
            unsetmask(cid);
        } else {
            shrink_mask_for_client(cid, need);
        }
    }

    vec.swap(new_vec);
}







// How long (µs) until this client can reach tpc_cor_opt TPCs,
// given its existing_tpcs and current running ops from *other* clients.
static inline double estimate_wait_time_for_knee(
    int tpc_cor_opt,
    int existing_tpcs,
    uint64_t now_ns,
    const std::vector<const op_info*>& running)
{
    // TPCs effectively available for this client *right now*
    int total_avail_now = num_tpcs + std::max(0, existing_tpcs);
    if (total_avail_now >= tpc_cor_opt) {
        return 0.0;  // no wait needed
    }

    struct Ev { uint64_t end; int tpc; };
    std::vector<Ev> events;
    events.reserve(running.size());

    for (const auto* r : running) {
        if (!r) continue;
        if (!r->is_running) continue;
        if (r->tpc_used <= 0) continue;
        events.push_back({r->est_finish_ns, r->tpc_used});
    }

    if (events.empty()) {
        // No one else running → num_tpcs should eventually cover tpc_cor_opt; just say no wait
        return 0.0;
    }

    std::sort(events.begin(), events.end(),
              [](const Ev& a, const Ev& b) { return a.end < b.end; });

    int freed = 0;
    for (const auto& e : events) {
        freed += e.tpc;
        int avail = total_avail_now + freed;
        if (avail >= tpc_cor_opt) {
            if (e.end <= now_ns) return 0.0;
            double wait_ns = double(e.end - now_ns);
            return wait_ns / 1000.0;  // ns -> µs
        }
    }

    // If still not enough (shouldn't happen if tpc_cor_opt <= HW TPC),
    // fallback: wait until the last one finishes.
    uint64_t last_end = events.back().end;
    if (last_end <= now_ns) return 0.0;
    return double(last_end - now_ns) / 1000.0;
}




// Predict co-run time for 'cur' at TPC = n, with 'running' as co-runners.
// Uses UT and pair ratios at *that* n.
// Returns time in microseconds (us), or NaN if cannot estimate.
static inline double corun_time_at_n(
    const op_info* cur,
    int n,
    const std::vector<const op_info*>& running)
{
    if (!cur) return std::numeric_limits<double>::quiet_NaN();

    if (n < 1) n = 1;
    if (n > num_tpcs) n = num_tpcs;

    // exclusive time at n
    float excl_n = excl_at_n(cur, n);
    if (!std::isfinite(excl_n) || excl_n <= 0.0f) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // UT ratio at this n
    double ut = ut_ratio(cur->cluster, n);
    double worst = ut;

    // worst pair ratio vs co-runners at this n
    for (const auto* r : running) {
        if (!r) continue;
        double pr  = pair_ratio(cur->cluster, r->cluster, n);
        double rco = std::max(pr, ut);
        if (rco > worst) worst = rco;
    }

    return excl_n * worst;  // us
}



static inline bool should_run_now_case3(
    const op_info* cur,
    int tpc_avail_now,                       // current available for this client
    uint64_t now_ns,                         // now
    const ProfileCheck& pc,                  // has TIME_COR_AVAIL, TPC_COR_OPT, etc.
    const std::vector<const op_info*>& running  // all running kernels (WT)
) {
    if (!cur) return false;
    if (tpc_avail_now <= 0) return false;
    if (!std::isfinite(pc.TIME_COR_AVAIL)) return false;

    const double T_now = pc.TIME_COR_AVAIL;  // predicted time if we run NOW with leftover TPCs

    // printf("[CASE3] check: T_now=TIME_COR_AVAIL=%.3f us, tpc_avail_now=%d, TPC_COR_OPT=%d\n",
    //        T_now, tpc_avail_now, pc.TPC_COR_OPT);

    // For each running kernel e ∈ WT:
    for (const op_info* r : running) {
        if (!r || !r->is_running || r->tpc_used <= 0) continue;

        // 1) wait time until this kernel finishes
        double wait_us = 0.0;
        if (r->est_finish_ns > now_ns) {
            wait_us = double(r->est_finish_ns - now_ns) / 1000.0;  // ns -> us
        }

        // 2) TPCs we could have if *this* kernel finishes
        int avail_after = tpc_avail_now + r->tpc_used;
        if (avail_after < 1) continue;

        // we don't need more than TPC_COR_OPT
        int n_use = std::min(avail_after, pc.TPC_COR_OPT);

        // 3) build running set *excluding* this e (since it's done when we start)
        std::vector<const op_info*> running_minus_e;
        running_minus_e.reserve(running.size());
        for (const op_info* x : running) {
            if (x == r) continue;
            if (x && x->is_running && x->tpc_used > 0) {
                running_minus_e.push_back(x);
            }
        }

        // 4) co-run time at n_use with remaining co-runners
        double T_cor_after = corun_time_at_n(cur, n_use, running_minus_e);
        if (!std::isfinite(T_cor_after)) {
            // printf("[CASE3]  e(cluster=%d, tpc=%d): cannot estimate corun at n=%d, skip.\n",
            //        r->cluster, r->tpc_used, n_use);
            continue;
        }

        double T_if_wait = wait_us + T_cor_after;

        // printf("[CASE3]  e(cluster=%d, tpc=%d): wait=%.3f us, n_use=%d, "
        //        "T_cor_after=%.3f us -> T_if_wait=%.3f vs T_now=%.3f\n",
        //        r->cluster, r->tpc_used, wait_us, n_use, T_cor_after, T_if_wait, T_now);

        // if *any* e gives us better/equal finish time by waiting, we should NOT run now
        if (T_if_wait <= T_now) {
            // printf("[CASE3]  found e with T_if_wait <= T_now, so do NOT run now.\n");
            return false;
        }
    }

    // For all e in WT: wait_e + T_cor(e at its n_use) > T_cor_avail  -> run now
    // printf("[CASE3]  for all e, wait_e + T_cor(e) > T_cor_avail -> run now.\n");
    return true;
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



// Toggle this to enable/disable estimation prints globally.
static bool g_estimator_verbose = true;

static inline double corun_time_at_n_exact_hw(
    const op_info* cur,
    int n,
    const op_info* co)
{
    if (!cur) {
        if (g_estimator_verbose) {
            std::printf("[EST] cur=null → return NaN\n");
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int n_in = n;

    // Clamp to hardware capacity (not dynamic leftover num_tpcs)
    if (n < 1) n = 1;
    if (n > HW_NUM_TPCS) n = HW_NUM_TPCS;

    if (g_estimator_verbose) {
        std::printf("[EST] begin: cur{id=%d name=%s cluster=%d} co{%s} n_in=%d → n_clamped=%d\n",
                    cur->id,
                    cur->name.c_str(),
                    cur->cluster,
                    (co ? ("id=" + std::to_string(co->id) + " name=" + co->name + " cluster=" + std::to_string(co->cluster)).c_str() : "null"),
                    n_in, n);
    }

    // Exclusive time at n
    float excl_n = excl_at_n(cur, n);
    if (!std::isfinite(excl_n) || excl_n <= 0.0f) {
        if (g_estimator_verbose) {
            std::printf("[EST] excl_at_n(cur,n=%d) invalid: %.6f → return NaN\n", n, excl_n);
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (g_estimator_verbose) {
        std::printf("[EST] excl_us_by_n[%d] = %.3f µs\n", n, excl_n);
    }

    // --- Pair-only multiplier (NO UT FALLBACK) ---
    double mult = 1.0;   // default: no slowdown if pair data missing
    double pr   = std::numeric_limits<double>::quiet_NaN();

    if (co) {
        pr = pair_ratio(cur->cluster, co->cluster, n);

        if (std::isfinite(pr) && pr > 0.0) {
            mult = pr;
            if (g_estimator_verbose) {
                std::printf("[EST] pair_ratio(UT=%d, CO=%d, n=%d) = %.6f (using PAIR ONLY)\n",
                            cur->cluster, co->cluster, n, pr);
            }
        } else {
            if (g_estimator_verbose) {
                std::printf("[EST] pair_ratio missing/invalid (%.6f) → use 1.0 (PAIR ONLY; no UT)\n", pr);
            }
            mult = 1.0;
        }
    } else {
        if (g_estimator_verbose) {
            std::printf("[EST] co=null → no pair available → use 1.0 (PAIR ONLY; no UT)\n");
        }
        mult = 1.0;
    }

    // Final estimate (pair-only)
    double est_us = excl_n * mult;
    if (g_estimator_verbose) {
        std::printf("[EST] result (PAIR ONLY): est_us = excl(%.3f) * pair_mult(%.6f) = %.3f µs\n",
                    excl_n, mult, est_us);
    }
    return est_us;  // µs
}



// duration_threshold = median of full-TPC exclusive latencies (in microseconds)
static inline double compute_duration_threshold_us() {
    std::vector<double> durations_us;
    durations_us.reserve(1024);  // arbitrary

    const int num_clients = (int)op_info_vector.size();
    for (int cid = 0; cid < num_clients; ++cid) {
        const auto& vec = op_info_vector[cid];
        for (size_t kid = 0; kid < vec.size(); ++kid) {
            const op_info* op = &vec[kid];

            // Try full TPCs first, then step down until we find a valid excl_at_n
            float best = NAN;
            for (int n = HW_NUM_TPCS; n >= 1; --n) {
                float v = excl_at_n(op, n);
                if (std::isfinite(v) && v > 0.f) {
                    best = v;
                    break;
                }
            }
            if (std::isfinite(best) && best > 0.f) {
                durations_us.push_back((double)best);
            }
        }
    }

    if (durations_us.empty()) {
        std::fprintf(stderr,
                     "[ERR] compute_duration_threshold_us: no valid excl_us_by_n values\n");
        std::abort();
    }

    std::sort(durations_us.begin(), durations_us.end());
    const size_t n = durations_us.size();

    double median;
    if (n & 1) {
        median = durations_us[n / 2];
    } else {
        median = 0.5 * (durations_us[n / 2 - 1] + durations_us[n / 2]);
    }

    std::printf("[INFO] duration_threshold (median full-TPC excl) = %.3f us (n=%zu)\n",
                median, n);
    return median;
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



//ficks
// ============================================================
// busy_wait_ficks (YOUR CODE) with ONLY minimal fixes:
//   FIX#1: op_info_vector index -> [client_id][seen[client_id]]
//   FIX#2: canSchedule check -> canSchedule[client_id] (NOT "if(canSchedule)")
// Everything else stays as close as possible to your original.
// ============================================================
void* Scheduler::busy_wait_ficks(int num_clients, int iter, bool warmup, int warmup_iters,
                                 bool seq, int depth, int hp_limit, int update_start)
{
    DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);

    depth = 5;
    using Clock = std::chrono::high_resolution_clock;
    auto start_total = Clock::now();

    while (1) {

        vector<func_record*> frecords(num_clients, NULL);

        for (int i = 0; i < num_clients; i++) {

            if (is_executing[i] == true) {
                continue;
            }

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
                }
            }
            pthread_mutex_unlock(client_mutexes[i]);
        }

        int num_all_clients = num_clients;
        vector<int> ready_client;

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
                if (frecords[j]->type != MALLOC_RECORD &&
                    frecords[j]->type != MEMCPY_RECORD &&
                    frecords[j]->type != MEMSET_RECORD &&
                    frecords[j]->type != FREE_RECORD &&
                    !warmup)
                {
                    ready_client.push_back(j);
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
        //   - build long/short queues
        //   - long first:
        //       * >=2 long ready -> schedule them
        //       * 1 long ready -> depth-lookahead in OTHER ACTIVE clients' next `depth` kernels:
        //             if any other has a long within depth, do NOT schedule this long now
        //             else schedule it (also schedule immediately if no other active client has work)
        //   - short after long -> schedule all short
        // ------------------------------------------------------------
        if (!warmup && !ready_client.empty()) {

            // Optional cleanup only (not a gate): if previous event finished, unsetmask
            for (int i = 0; i < num_clients; ++i) {
                if (event_ids[i] >= 1) {
                    if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                        unsetmask(i);
                    }
                }
            }

            std::vector<int> long_q;
            std::vector<int> short_q;
            long_q.reserve(ready_client.size());
            short_q.reserve(ready_client.size());

            for (int cid : ready_client) {
                const op_info& op = op_info_vector[cid][seen[cid]];
                if (op.is_short == 1) short_q.push_back(cid);
                else                  long_q.push_back(cid);
            }

            // -------------------------
            // 1) LONG first
            // -------------------------
            if (long_q.size() >= 2) {
            // if (long_q.size() >= ) {

                // schedule all currently-ready longs (your current behavior)
                for (int cid : long_q) {
                    schedule_kernel(*(frecords[cid]),
                                    sched_streams[cid],
                                    cid,
                                    events[cid][event_ids[cid]],
                                    seen, event_ids, cid);
                    pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);
                }

            }
            else if (long_q.size() == 1) {

                int cid = long_q[0];

                // deadlock guard: if no other ACTIVE client has remaining kernels, schedule now
                bool other_has_work = false;
                for (int j = 0; j < num_clients; ++j) {
                    if (j == cid) continue;

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
                    schedule_kernel(*(frecords[cid]),
                                    sched_streams[cid],
                                    cid,
                                    events[cid][event_ids[cid]],
                                    seen, event_ids, cid);
                    pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);
                }
                else {

                    // -------- depth lookahead (your requested style) --------
                    int look = depth;
                    if (look < 0) look = 0;

                    bool found_other_long_within_depth = false;

                    // First find other clients that "have work" (active + unfinished),
                    // then scan their next `look` kernels via op_info_vector.
                    if (look > 0) {
                        for (int j = 0; j < num_clients; ++j) {
                            if (j == cid) continue;

                            // "client has work" condition
                            if ((num_client_cur_iters[j] == num_client_max_iters[j]) ||
                                (warmup && (num_client_cur_iters[j] == warmup_iters)) ||
                                (stop_ack[j] == true))
                                continue;

                            if (seen[j] >= num_client_kernels[j]) continue;

                            // scan next `look` kernels
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

                    // If we found another long within depth, we simply DON'T schedule this single long now.
                    // Otherwise, schedule it.
                    if (!found_other_long_within_depth) {
                        schedule_kernel(*(frecords[cid]),
                                        sched_streams[cid],
                                        cid,
                                        events[cid][event_ids[cid]],
                                        seen, event_ids, cid);
                        pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);
                    }
                    // else: do nothing this round (wait)
                }
            }

            // -------------------------
            // 2) SHORT after long
            // -------------------------
            if (!short_q.empty()) {
                for (int cid : short_q) {
                    schedule_kernel(*(frecords[cid]),
                                    sched_streams[cid],
                                    cid,
                                    events[cid][event_ids[cid]],
                                    seen, event_ids, cid);
                    pop_from_queue(client_buffers[cid], client_mutexes[cid], cid);
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
                // check if GPU work for this client has finished
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

                    // reset meta-structures for this client, and let it continue
                    seen[i] = 0;
                    if (seq)
                        event_ids[0] = 0;
                    event_ids[i] = 0;
                    streams[i] = -1;
                    fidx[i] = 0;
                    request_status[i][num_client_cur_iters[i]] = true;

                    pthread_mutex_unlock(client_mutexes[i]);
                    num_client_cur_iters[i] += 1;
                    locked[i] = false;
                    client_progress[i] = 0;

                    auto end = std::chrono::high_resolution_clock::now();
                    float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
                    duration /= 1000.0;
                    client_durations[i].push_back(duration);
                }
            }
        }

        if (finished == num_clients)
            break;
    }

    if (!warmup) {
        auto end_total = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
        duration /= 1000.0;
        printf("Total loop took %ld nanoseconds\n", duration);
        // process_eval(client_durations);
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
                // info.opt_tpc_colocated = std::stoi(v[3]);
                // info.opt_tpc_exclusive = std::stoi(v[4]);
                info.is_short = std::stoi(v[3]);
                info.sm_used           = 0;
                info.tpc_used          = 0;

                info.excl_us_by_n.clear();
                info.excl_us_by_n.push_back(
                    std::numeric_limits<float>::quiet_NaN());  // index 0 unused

                for (size_t i = 4; i < v.size(); ++i) {
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
        ){

        g_TD1 = td1;
    	g_TD2 = td2;
		for (int i = 0; i < num_clients; i++) {
        model_names.emplace_back(models[i]);  // convert char* → std::string
		}

		// Example: print them
		for (const auto& name : model_names) {
			std::cout << "Model: " << name << std::endl;
		}


        const char* RATIO_ROOT = "/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/ficks/profiling_old/";
		load_ratio_tables_once(RATIO_ROOT);

		struct passwd *pw = getpwuid(getuid());
		char *homedir = pw->pw_dir;
		// const char* lib_path = "/orion_bu/src/cuda_capture/libinttemp.so";
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

		// 1. thread structures
		pid_t** thread_ids_all = (pid_t**)dlsym(klib, "thread_ids");
		*thread_ids_all = (pid_t*)malloc((2*num_clients+1)*sizeof(pid_t)); // 2*N threads + scheduler

		for (int i=0; i<num_clients; i++)
			(*thread_ids_all)[i] = tids[i];
		(*thread_ids_all)[num_clients] = mytid;
		for (int i=num_clients+1; i<2*num_clients+1; i++)
			(*thread_ids_all)[i] = 0;
		//printf("address is %p, %p\n", thread_ids_all, *thread_ids_all);

		int** num_total_clients = (int**)dlsym(klib, "num_total_clients");
		*num_total_clients = (int*)malloc(sizeof(int));
		**num_total_clients = num_clients;

		num_cur_clients.resize(num_clients);
		is_executing.resize(num_clients);
		for (int i = 0; i < num_clients; ++i) {
			num_cur_clients[i] = i;
			is_executing[i] = false;
		}
		client_finished = new bool[num_clients](); // Initialize all elements to false

		for (int i=0; i<=num_clients; i++) {
			DEBUG_PRINT("Scheduler setup the thread id at %d to be %d\n", i, (*thread_ids_all)[i]);
		}

        // penalty.assign(num_clients, 0);  // resize & zero-init
        
		// 2. metadata structures
		for (int i=0; i<num_clients; i++) {
			op_info_vector.push_back({});
			client_durations.push_back({});
			populate_kernel_info(files[i], op_info_vector[i],algo_id);
			int max_sm_used = 0;
			for (auto info: op_info_vector[i])
				max_sm_used = max(max_sm_used, info.sm_used);
			max_sms_clients.push_back(max_sm_used);
			printf("----------- SIZE: %ld\n", op_info_vector[i].size());
			is_train.push_back(train[i]);
			client_progress.push_back(0);
			func_progress.push_back(-1);
		}

		// 3. indexes
		int** fidx_ptr = (int**)dlsym(klib, "func_indexes");
		*fidx_ptr = (int*)calloc(num_clients, sizeof(int));
		fidx = *fidx_ptr;

		num_client_kernels = num_kernels;
		num_client_max_iters = num_iters;

		num_client_cur_iters = (int*)calloc(num_clients, sizeof(int));
		locked = (bool*)calloc(num_clients, sizeof(bool));

		// to get measurements
		client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		total_client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		client_starts_set = (bool**)malloc(num_clients*sizeof(bool*));
		for (int i=0; i<num_clients; i++) {
			client_starts_set[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
		}

		// 4. communication queues + locks
		queue<func_record>*** buffers_ptr = (queue<func_record>***)dlsym(klib, "kqueues");
		*buffers_ptr = (queue<func_record>**)malloc(num_clients*sizeof(queue<func_record>*));
		queue<func_record>** buffers = *buffers_ptr;
		for (int i=0; i<num_clients; i++) {
			buffers[i] = new queue<func_record>();
			printf("buffer size is %ld\n", buffers[i]->size());
		}

		pthread_mutex_t*** client_mutexes_ptr = (pthread_mutex_t***)dlsym(klib, "mutexes");
		*client_mutexes_ptr = (pthread_mutex_t**)malloc(num_clients*sizeof(pthread_mutex_t*));
		client_mutexes = *client_mutexes_ptr;
		for (int i=0; i<num_clients; i++) {
			client_mutexes[i] = new pthread_mutex_t(); //(pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		}

        if(algo_id == ALGO_REEF){
            scheduler->profile_prep(buffers, num_clients, 1);
        }
        else{
            scheduler->profile_prep(buffers, num_clients, 0);
        }
		

		// 5. runtime control
		bool*** request_status_ptr = (bool***)dlsym(klib, "client_request_status");
		*request_status_ptr = (bool**)malloc(num_clients*sizeof(bool*));
		request_status = *request_status_ptr;

		// check!
		bool** stops_ptr = (bool**)dlsym(klib, "client_stop");
		*stops_ptr = (bool*)calloc(num_clients, sizeof(bool));
		stops = *stops_ptr;

		bool** stop_ack_ptr = (bool**)dlsym(klib, "client_stop_ack");
		*stop_ack_ptr = (bool*)calloc(num_clients, sizeof(bool));
		stop_ack = *stop_ack_ptr;

		bool** affinity_set_ptr = (bool**)dlsym(klib, "affinity_set");
		(*affinity_set_ptr) = (bool*)calloc(num_clients+1, sizeof(bool));

		for (int i=0; i<num_clients; i++) {
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