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
std::vector<RunningOp> g_running_ops;

// [UT][CO][n] -> multiplier (default 1.0)
static std::vector<std::vector<std::vector<double>>> g_pair;
// [UT][n]     -> multiplier (default 1.0)
static std::vector<std::vector<double>> g_ut;
static bool g_loaded = false;
static float g_TD1 = 1.2f;
static float g_TD2 = 1.1f;
static constexpr int HW_NUM_TPCS = 24;

// ---- O(1) lookups (ABORT if missing, unless use_excl) ----

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





static std::string csv_escape(const std::string& s_in) {
    std::string s = s_in;
    size_t pos = 0;
    while ((pos = s.find('\"', pos)) != std::string::npos) {
        s.insert(pos, 1, '\"'); // double the quote
        pos += 2;
    }
    return "\"" + s + "\"";
}

// simple file-exists helper
static bool file_exists(const std::string& path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0);
}

// optional: make sure dir exists (no-op if already there)
static bool ensure_dir_exists(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) return true;
        fprintf(stderr, "ERROR: %s exists but is not a directory.\n", dir.c_str());
        return false;
    }
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "ERROR: mkdir(%s) failed (errno=%d).\n", dir.c_str(), errno);
        return false;
    }
    return true;
}

// base_prefix: e.g. "reef_kernel_schedule_log"
// outputs: <out_dir>/<base_prefix>_client<id>.csv, or with _1, _2, ... if already exists
static void write_kernel_logs_per_client(
    const std::string& out_dir,
    const std::string& base_prefix,
    const std::vector<Scheduler::KernelLogEntry>& logs)
{
    if (logs.empty()) {
        fprintf(stderr, "write_kernel_logs_per_client: no logs, nothing to write.\n");
        return;
    }

    if (!ensure_dir_exists(out_dir)) {
        fprintf(stderr, "write_kernel_logs_per_client: output dir '%s' not usable.\n",
                out_dir.c_str());
        return;
    }

    // ----------------------------
    // IMPORTANT FIX:
    // Use ONE global minimum start_ns across ALL logs
    // so timestamps are comparable between client files.
    // ----------------------------
    long long global_min_start = logs[0].start_ns;
    for (const auto& e : logs) {
        if (e.start_ns < global_min_start) {
            global_min_start = e.start_ns;
        }
    }

    // Group entries by client_id
    std::unordered_map<int, std::vector<const Scheduler::KernelLogEntry*>> by_client;
    by_client.reserve(16);

    for (const auto& e : logs) {
        by_client[e.client_id].push_back(&e);
    }

    for (auto& kv : by_client) {
        int client_id = kv.first;
        auto& vec     = kv.second;
        if (vec.empty()) continue;

        // Build base file name:
        //   <out_dir>/<base_prefix>_client<id>.csv
        std::string base_name;
        {
            char tmp[256];
            std::snprintf(tmp, sizeof(tmp),
                          "%s_client%d.csv",
                          base_prefix.c_str(), client_id);
            base_name = out_dir + "/" + tmp;
        }

        // If exists, append _1, _2, ...
        std::string final_name = base_name;
        int suffix = 1;
        while (file_exists(final_name)) {
            char tmp[256];
            std::snprintf(tmp, sizeof(tmp),
                          "%s_client%d_%d.csv",
                          base_prefix.c_str(), client_id, suffix);
            final_name = out_dir + "/" + tmp;
            ++suffix;
        }

        std::ofstream ofs(final_name);
        if (!ofs.good()) {
            fprintf(stderr, "ERROR: cannot open %s for writing.\n", final_name.c_str());
            continue;
        }

        // Header: add model_name right after kernel_name (or wherever you prefer)
        ofs << "client_id,iter,kernel_id,event_id,tpc_used,kernel_name,model_name,"
               "start_us,end_us,duration_us\n";

        for (const auto* e : vec) {

            // GLOBAL aligned axis: shift by global_min_start (not per-client)
            long long start_rel_ns = e->start_ns - global_min_start;
            long long end_rel_ns   = (e->end_ns > 0) ? (e->end_ns - global_min_start) : 0;

            long long start_us    = start_rel_ns / 1000;      // ns -> us
            long long end_us      = end_rel_ns   / 1000;      // ns -> us
            long long duration_us = e->duration_ns / 1000;    // ns -> us

            ofs << e->client_id   << ','
                << e->iter_id     << ','
                << e->kernel_id   << ','
                << e->event_id    << ','
                << e->tpc_used    << ','
                << csv_escape(e->kernel_name) << ','
                << csv_escape(e->model_name)  << ','   // <----- ADDED
                << start_us       << ','
                << end_us         << ','
                << duration_us    << '\n';
        }

        ofs.close();
        printf("Per-client log written: %s (%zu rows)\n",
               final_name.c_str(), vec.size());
    }
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
	localMask_O= (uint32_t*)calloc(num,sizeof(uint32_t));
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
            setmask( tpc_usage, hp_client);
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
                                int update_start){
    // printf("Entered busy_wait_profile! (ORION) Num clients is %d\n", num_clients);

    int start0 = 0;
    int start1 = 0;

    int prev_large  = -1;
    int hp_running  = -1;

    bool inf_finished = false;
    bool started      = false;

    auto start_total = std::chrono::high_resolution_clock::now();

    std::vector<bool> total_client_set(num_clients, false);
    std::vector<int>  profiles(num_clients, -1);
    std::vector<int>  cur_sms(num_clients, -1);

    int   hp_client        = num_clients - 1;
    bool  large_found      = false;
    long  sum              = 0;
    long  size             = 0;
    int   start            = -1;

    int   low_sms          = 0;
    int   high_sms         = max_sms_clients[0]; // 0 is the LP client
    int   sm_threshold     = max_sms_clients[0] / 2;
    float hp_iter_duration = 0.0f;
    float hp_limit_float   = (float)hp_limit;
    int   coexe_step       = 0;

    if (!is_train[hp_client]) {
        sm_threshold = max_sms;
        update_start = INT_MAX;
    }

    int      SM_hp_client = 0;
    int      total_SM     = max_sms;
    long int endT         = 0;

    // =========================
    // Logging (ns scale; mem/admin not logged)
    // =========================

    auto to_ns = [](const std::chrono::high_resolution_clock::time_point& t0,
                    const std::chrono::high_resolution_clock::time_point& t) -> long long {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count();
    };

    std::vector<Scheduler::KernelLogEntry>               kernel_logs;
    std::vector<std::unordered_map<int, int>> event2log(num_clients);

    auto close_event_log = [&](int client, int ev_done) {
        auto it = event2log[client].find(ev_done);
        if (it != event2log[client].end()) {
            Scheduler::KernelLogEntry& e = kernel_logs[it->second];
            if (e.end_ns == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                e.end_ns      = to_ns(start_total, now);
                e.duration_ns = (e.end_ns - e.start_ns);
            }
            event2log[client].erase(it);
        }
    };

    // Optional debug helper (no-op)
    auto dump_state = [&](const char* /*phase_tag*/) {
        // previously printed internal state; now no-op
    };

    // ======================================================
    // SINGLE-PHASE: ORION HP/BE SCHEDULER OVER FULL RUN
    // ======================================================
    while (1) {
        dump_state("ORION");

        std::vector<func_record*> frecords(num_clients, NULL);

        // Pull one record per client (same as existing code, minus window checks)
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
                        total_client_set[i]    = true;
                    }
                }
            }
            pthread_mutex_unlock(client_mutexes[i]);
        }

        bool canSchedule[num_clients];
        for (int i = 0; i < num_clients; ++i) {
            canSchedule[i] = true;
            if (event_ids[i] >= 1) {
                if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                    int ev_done = event_ids[i] - 1;
                    close_event_log(i, ev_done);
                    // ORION uses TPC masks: unset when kernel finishes
                    unsetmask(i);
                } else {
                    canSchedule[i] = false;
                }
            }
        }

        bool schedule_BE = false;
        endT             = 0;

        // ---- HP client scheduling ----
        if (frecords[hp_client] != NULL) {
            if (frecords[hp_client]->type != MALLOC_RECORD &&
                frecords[hp_client]->type != MEMCPY_RECORD &&
                frecords[hp_client]->type != MEMSET_RECORD &&
                frecords[hp_client]->type != FREE_RECORD) {

                if (canSchedule[hp_client] == false) {
                    // cannot schedule HP this round
                    goto after_hp;
                }

                op_info op_info_1 = op_info_vector[hp_client][seen[hp_client]];
                int     tpc_usage = op_info_1.sm_used / 2;
                tpc_usage         = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;

                if (num_tpcs >= tpc_usage) {
                    setmask(tpc_usage, hp_client);

                    if (!warmup) {
                        int  ev_id_before    = event_ids[hp_client];
                        auto now             = std::chrono::high_resolution_clock::now();
                        int  kernel_id_before = seen[hp_client];
                        std::string kname    = op_info_vector[hp_client][kernel_id_before].name;

                        Scheduler::KernelLogEntry entry{
                            hp_client,
                            num_client_cur_iters[hp_client],
                            kernel_id_before,
                            ev_id_before,
                            tpc_usage,
                            kname,
                            "",
                            to_ns(start_total, now),
                            0,
                            0
                        };


                        event2log[hp_client][ev_id_before] = (int)kernel_logs.size();
                        kernel_logs.push_back(entry);
                    }

                    schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client],
                                    hp_client, events[hp_client][event_ids[hp_client]],
                                    seen, event_ids, hp_client);

                    pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);

                    int   dur       = op_info_1.duration;
                    float threshold = 0.8f;
                    auto  current_time =
                        std::chrono::high_resolution_clock::now();
                    auto duration_ns =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            current_time.time_since_epoch()).count();
                    endT = static_cast<long int>(
                               static_cast<double>(dur) * threshold +
                               static_cast<double>(duration_ns));
                    schedule_BE = true;
                }
            } else {
                // HP mem/admin
                schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client],
                                hp_client, events[hp_client][event_ids[hp_client]],
                                seen, event_ids, hp_client);
                pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
            }
        } else {
            hp_client = (hp_client + 1) % num_clients;
        }

    after_hp:

        // ---- BE clients scheduling ----
        if (schedule_BE) {
            for (int t = 1; t < num_clients; t++) {
                int j = (hp_client + t) % num_clients;
                if (frecords[j] != NULL) {
                    op_info op_info_0 = op_info_vector[j][seen[j]];
                    int     tpc_usage = op_info_0.sm_used / 2;
                    tpc_usage         = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;
                    bool    schedule  = false;

                    if ((num_clients == 1) || (seen[hp_client] == 0) ||
                        (frecords[j]->type == MALLOC_RECORD) ||
                        (frecords[j]->type == MEMCPY_RECORD) ||
                        (frecords[j]->type == MEMSET_RECORD) ||
                        (frecords[j]->type == FREE_RECORD)) {
                        schedule = true;
                    }
                    else if (num_tpcs > 0 && seen[hp_client] > 0 &&
                             ((op_info_0.profile == -1 || profiles[hp_client] == -1 ||
                               (profiles[hp_client] != op_info_0.profile)))) {
                        auto current_time =
                            std::chrono::high_resolution_clock::now();
                        auto duration_ns =
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                current_time.time_since_epoch()).count();
                        if (endT == 0) {
                            schedule = true;
                        } else if (duration_ns + op_info_0.duration <= endT) {
                            schedule = true;
                        }
                    }

                    if (schedule) {
                        int assigned_tpc = 0;

                        if (frecords[j]->type != MALLOC_RECORD &&
                            frecords[j]->type != MEMCPY_RECORD &&
                            frecords[j]->type != MEMSET_RECORD &&
                            frecords[j]->type != FREE_RECORD) {

                            if (canSchedule[j] == false) {
                                continue;
                            }

                            assigned_tpc = std::min(num_tpcs, tpc_usage);
                            setmask(assigned_tpc, j);

                            if (!warmup) {
                                int  ev_id_before    = event_ids[j];
                                auto now             = std::chrono::high_resolution_clock::now();
                                int  kernel_id_before = seen[j];
                                std::string kname    =
                                    (kernel_id_before >= 0 &&
                                     kernel_id_before < (int)op_info_vector[j].size())
                                        ? op_info_vector[j][kernel_id_before].name
                                        : "UNKNOWN";

                                Scheduler::KernelLogEntry entry{
                                    j,
                                    num_client_cur_iters[j],
                                    kernel_id_before,
                                    ev_id_before,
                                    assigned_tpc,
                                    kname,
                                    "",
                                    to_ns(start_total, now),
                                    0,
                                    0
                                };
                                event2log[j][ev_id_before] = (int)kernel_logs.size();
                                kernel_logs.push_back(entry);
                            }
                        }

                        schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                        events[j][event_ids[j]], seen, event_ids, j);
                        pop_from_queue(client_buffers[j], client_mutexes[j], j);
                    }
                }
            }
            hp_client = (hp_client + 1) % num_clients;
        } else {
            // no BE co-exec; still schedule mem/admin from other clients
            for (int t = 1; t < num_clients; t++) {
                int j = (hp_client + t) % num_clients;
                if (frecords[j] != NULL) {
                    if (frecords[j]->type == MALLOC_RECORD ||
                        frecords[j]->type == MEMCPY_RECORD ||
                        frecords[j]->type == MEMSET_RECORD ||
                        frecords[j]->type == FREE_RECORD) {

                        schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                        events[j][event_ids[j]], seen, event_ids, j);
                        pop_from_queue(client_buffers[j], client_mutexes[j], j);
                    }
                }
            }
        }

        // ---- Finish detection / per-iter bookkeeping ----
        int finished = 0;
        for (int i = 0; i < num_clients; i++) {
            if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
                (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
                (stop_ack[i] == true)) {
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
                    float duration_ms =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count()
                        / 1000.0f;
                    client_durations[i].push_back(duration_ms);
                }
            }
        }

        if (finished == num_clients) {
            break;
        }
    }

    if (!warmup) {
        auto end_total = std::chrono::high_resolution_clock::now();
        float duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count()
            / 1000.0f;
        (void)duration;   // silence unused if you don't print it
        (void)coexe_step; // silence unused
    }

    if (!warmup) {
        // Close remaining events as you already do...
        for (int i = 0; i < num_clients; ++i) {
            if (event_ids[i] >= 1) {
                int ev_last = event_ids[i] - 1;
                close_event_log(i, ev_last);
            }
        }

        // One CSV per client, time shifted to start at 0
        write_kernel_logs_per_client("kernel_logs", "orion_kernel_schedule_log", kernel_logs);
    }


    return NULL;
}





// KRISP
void* Scheduler::busy_wait_krisp(int num_clients, int iter, bool warmup, int warmup_iters,
                                    bool seq, int depth, int hp_limit, int update_start)
{
    DEBUG_PRINT("Entered busy_wait_profile (masked scheduler)! Num clients is %d\n", num_clients);

    int start0 = 0;
    int start1 = 0;

    auto start_total = std::chrono::high_resolution_clock::now();

    std::vector<bool> total_client_set(num_clients, false);
    std::vector<int>  profiles(num_clients, -1);
    std::vector<int>  cur_sms(num_clients, -1);

    int hp_client = 1;
    int lp_client = 0;

    bool large_found = false;
    long sum  = 0;   // sum of durations of ongoing BE kernels
    long size = 0;   // sum of sizes of in-the-queues BE kernels
    int  start = -1;

    // BS - works only for 2 clients for now
    int   low_sms        = 0;
    int   high_sms       = max_sms_clients[0]; // 0 is the lp client
    int   sm_threshold   = max_sms_clients[0] / 2;
    float hp_iter_duration = 0.0f; // 1 is the hp client
    float hp_limit_float   = (float)hp_limit;

    auto to_ns = [](const std::chrono::high_resolution_clock::time_point& t0,
                    const std::chrono::high_resolution_clock::time_point& t) -> long long {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count();
    };

    // Helper: pick "knee TPC" from op_info; clamp [1, num_tpcs]
    auto pick_knee_tpc = [&](const op_info& op) -> int {
        int knee = op.knee_tpc;  // <- make sure this exists in op_info
        if (knee < 1) knee = 1;
        if (knee > num_tpcs) knee = num_tpcs;
        return knee;
    };

    std::vector<Scheduler::KernelLogEntry>               kernel_logs;
    std::vector<std::unordered_map<int, int>> event2log(num_clients); // per-client: event_id -> index in kernel_logs


    auto close_event_log = [&](int client, int ev_done) {
        auto it = event2log[client].find(ev_done);
        if (it != event2log[client].end()) {
            Scheduler::KernelLogEntry& e = kernel_logs[it->second];
            if (e.end_ns == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                e.end_ns      = to_ns(start_total, now);
                e.duration_ns = (e.end_ns - e.start_ns);
            }
            event2log[client].erase(it);
        }
    };

    while (1) {
        std::vector<func_record*> frecords(num_clients, NULL);
        size = 0;

        // ----------------------------
        // Pull one record per client
        // ----------------------------
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

        bool canSchedule[num_clients];

        // Close finished events -> fill end_ns/duration_ns AND UNSET MASK
        for (int i = 0; i < num_clients; ++i) {
            canSchedule[i] = true;
            if (event_ids[i] >= 1) {
                if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                    auto now    = std::chrono::high_resolution_clock::now();
                    int  ev_done = event_ids[i] - 1;
                    auto it      = event2log[i].find(ev_done);
                    if (it != event2log[i].end()) {
                        Scheduler::KernelLogEntry& e = kernel_logs[it->second];
                        if (e.end_ns == 0) {
                            e.end_ns      = to_ns(start_total, now);
                            e.duration_ns = (e.end_ns - e.start_ns);
                        }
                        event2log[i].erase(it);
                    }
                    // IMPORTANT: unset mask when a kernel finishes
                    unsetmask(i);
                } else {
                    canSchedule[i] = false;
                }
            }
        }

        int num_all_clients = num_clients;
        for (int i = 0; i < num_clients; i++) {
            if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
                (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
                (stop_ack[i] == true)) {
                num_all_clients -= 1;
            }
        }

        // ------------------------------------------
        // Schedule everything available
        // ------------------------------------------
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
                int     knee_tpc   = pick_knee_tpc(opj);
                int     assigned_tpc = min(num_tpcs, knee_tpc);
                if (assigned_tpc < 1) continue; // safety

                // 3) Set the mask per your hint
                setmask(assigned_tpc, j);


                // 4) Schedule
                schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                events[j][event_ids[j]], seen, event_ids, j);
                pop_from_queue(client_buffers[j], client_mutexes[j], j);
           


                // 2) Logging: only if NOT warmup
                if (!warmup) {
                    int ev_id_before      = event_ids[j];
                    auto now              = std::chrono::high_resolution_clock::now();
                    int  kernel_id_before = seen[j];
                    std::string kname =
                        (kernel_id_before >= 0 && kernel_id_before < (int)op_info_vector[j].size())
                        ? op_info_vector[j][kernel_id_before].name
                        : "UNKNOWN";

                    Scheduler::KernelLogEntry entry{
                        /*client_id*/   j,
                        /*iter_id  */   num_client_cur_iters[j],
                        /*kernel_id*/   kernel_id_before,
                        /*event_id */   ev_id_before,
                        /*tpc_used*/    assigned_tpc,     // record mask size we will set
                        /*kernel_name*/ kname,
                                        "",
                        /*start_ns*/    to_ns(start_total, now),
                        /*end_ns  */    0,
                        /*duration*/    0
                    };
                    event2log[j][ev_id_before] = (int)kernel_logs.size();
                    kernel_logs.push_back(entry);
                }

                // 3) Set the mask per your hint
                // setmask(assigned_tpc, j);

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

        int finished = 0;
        for (int i = 0; i < num_clients; i++) {

            if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
                (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
                (stop_ack[i] == true)) {
                finished += 1;
            } else if (seen[i] == num_client_kernels[i]) {
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
                } else {
                    if (event_ids[i] >= 1) {
                        if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess)
                            ready &= false;
                    }
                }
                if (ready) {
                    // IMPORTANT: unset mask here too when client's batch finishes
                    unsetmask(i);

                    // reset meta-structures for this client, and let it continue
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
                    client_progress[i] = 0;
                    auto end = std::chrono::high_resolution_clock::now();
                    float duration_ms =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count()
                        / 1000.0f;
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

    if (!warmup) {
        // Close remaining events as you already do...
        for (int i = 0; i < num_clients; ++i) {
            if (event_ids[i] >= 1) {
                int ev_last = event_ids[i] - 1;
                close_event_log(i, ev_last);
            }
        }

        // One CSV per client, time shifted to start at 0
        write_kernel_logs_per_client("kernel_logs", "krisp_kernel_schedule_log", kernel_logs);
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

    // =========================
    // Local logging (ns scale; mem/admin not logged)
    // =========================
    auto to_ns = [](const std::chrono::high_resolution_clock::time_point& t0,
                    const std::chrono::high_resolution_clock::time_point& t)
                 -> long long {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count();
    };

    std::vector<Scheduler::KernelLogEntry>                kernel_logs;
    std::vector<std::unordered_map<int,int>>   event2log(num_clients); // event_id -> index in kernel_logs

    // Helper: update log when an event completes
    // (Only compute kernels have entries in event2log, because we only push logs when !warmup)
    auto close_event_log = [&](int client, int ev_done) {
        auto it = event2log[client].find(ev_done);
        if (it != event2log[client].end()) {
            auto now = std::chrono::high_resolution_clock::now();
            Scheduler::KernelLogEntry& e = kernel_logs[it->second];

            if (e.end_ns == 0) {
                e.end_ns      = to_ns(start_total, now);
                e.duration_ns = e.end_ns - e.start_ns;
            }

            event2log[client].erase(it);
        }
    };

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
                    int ev_done = event_ids[i] - 1;

                    // finish update only exists for compute kernels when !warmup
                    close_event_log(i, ev_done);

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

                        // LOG only when !warmup
                        if (!warmup) {
                            int ev_id_before      = event_ids[hp_client];
                            auto now              = std::chrono::high_resolution_clock::now();
                            int  kernel_id_before = seen[hp_client];

                            std::string kname =
                                (kernel_id_before >= 0 &&
                                 kernel_id_before < (int)op_info_vector[hp_client].size())
                                ? op_info_vector[hp_client][kernel_id_before].name
                                : "UNKNOWN";

                            Scheduler::KernelLogEntry entry{
                                /*client_id*/   hp_client,
                                /*iter_id  */   num_client_cur_iters[hp_client],
                                /*kernel_id*/   kernel_id_before,
                                /*event_id */   ev_id_before,
                                /*tpc_used*/    tpc_usage,
                                /*kernel_name*/ kname,
                                                "",
                                /*start_ns*/    to_ns(start_total, now),
                                /*end_ns  */    0,
                                /*duration*/    0
                            };
                            event2log[hp_client][ev_id_before] = (int)kernel_logs.size();
                            kernel_logs.push_back(entry);
                        }

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

                            // LOG only when !warmup
                            if (!warmup) {
                                int ev_id_before      = event_ids[i];
                                auto now              = std::chrono::high_resolution_clock::now();
                                int  kernel_id_before = seen[i];

                                std::string kname =
                                    (kernel_id_before >= 0 &&
                                     kernel_id_before < (int)op_info_vector[i].size())
                                    ? op_info_vector[i][kernel_id_before].name
                                    : "UNKNOWN";

                                Scheduler::KernelLogEntry entry{
                                    /*client_id*/   i,
                                    /*iter_id  */   num_client_cur_iters[i],
                                    /*kernel_id*/   kernel_id_before,
                                    /*event_id */   ev_id_before,
                                    /*tpc_used*/    assigned,
                                    /*kernel_name*/ kname,
                                                    "",
                                    /*start_ns*/    to_ns(start_total, now),
                                    /*end_ns  */    0,
                                    /*duration*/    0
                                };
                                event2log[i][ev_id_before] = (int)kernel_logs.size();
                                kernel_logs.push_back(entry);
                            }

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

                                // LOG only when !warmup
                                if (!warmup) {
                                    int ev_id_before      = event_ids[i];
                                    auto now              = std::chrono::high_resolution_clock::now();
                                    int  kernel_id_before = seen[i];

                                    std::string kname =
                                        (kernel_id_before >= 0 &&
                                         kernel_id_before < (int)op_info_vector[i].size())
                                        ? op_info_vector[i][kernel_id_before].name
                                        : "UNKNOWN";

                                    Scheduler::KernelLogEntry entry{
                                        /*client_id*/   i,
                                        /*iter_id  */   num_client_cur_iters[i],
                                        /*kernel_id*/   kernel_id_before,
                                        /*event_id */   ev_id_before,
                                        /*tpc_used*/    assigned,
                                        /*kernel_name*/ kname,
                                                        "",
                                        /*start_ns*/    to_ns(start_total, now),
                                        /*end_ns  */    0,
                                        /*duration*/    0
                                    };
                                    event2log[i][ev_id_before] = (int)kernel_logs.size();
                                    kernel_logs.push_back(entry);
                                }

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

    // =========================
    // Write CSVs (only if !warmup)
    // =========================
    if (!warmup) {
        // Close remaining events (only compute events exist in event2log)
        for (int i = 0; i < num_clients; ++i) {
            if (event_ids[i] >= 1) {
                int ev_last = event_ids[i] - 1;
                close_event_log(i, ev_last);
            }
        }

        write_kernel_logs_per_client("kernel_logs", "reef_kernel_schedule_log", kernel_logs);
    }

    return NULL;
}



// Multiple stream
// void* Scheduler::busy_wait_ms(int num_clients, int iter, bool warmup, int warmup_iters,
//                                     bool seq, int depth, int hp_limit, int update_start)
// {
//     DEBUG_PRINT("Entered busy_wait_profile (masked scheduler)! Num clients is %d\n", num_clients);

//     int start0 = 0;
//     int start1 = 0;

//     auto start_total = std::chrono::high_resolution_clock::now();

//     std::vector<bool> total_client_set(num_clients, false);
//     std::vector<int>  profiles(num_clients, -1);
//     std::vector<int>  cur_sms(num_clients, -1);

//     int hp_client = 1;
//     int lp_client = 0;

//     bool large_found = false;
//     long sum  = 0;   // sum of durations of ongoing BE kernels
//     long size = 0;   // sum of sizes of in-the-queues BE kernels
//     int  start = -1;

//     // BS - works only for 2 clients for now
//     int   low_sms        = 0;
//     int   high_sms       = max_sms_clients[0]; // 0 is the lp client
//     int   sm_threshold   = max_sms_clients[0] / 2;
//     float hp_iter_duration = 0.0f; // 1 is the hp client
//     float hp_limit_float   = (float)hp_limit;

//     auto to_ns = [](const std::chrono::high_resolution_clock::time_point& t0,
//                     const std::chrono::high_resolution_clock::time_point& t) -> long long {
//         return std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count();
//     };

//     // Helper: pick "knee TPC" from op_info; clamp [1, num_tpcs]
//     auto pick_knee_tpc = [&](const op_info& op) -> int {
//         int knee = op.knee_tpc;  // <- make sure this exists in op_info
//         if (knee < 1) knee = 1;
//         if (knee > num_tpcs) knee = num_tpcs;
//         return knee;
//     };

//     std::vector<Scheduler::KernelLogEntry>               kernel_logs;
//     std::vector<std::unordered_map<int, int>> event2log(num_clients); // per-client: event_id -> index in kernel_logs


//     auto close_event_log = [&](int client, int ev_done) {
//         auto it = event2log[client].find(ev_done);
//         if (it != event2log[client].end()) {
//             Scheduler::KernelLogEntry& e = kernel_logs[it->second];
//             if (e.end_ns == 0) {
//                 auto now = std::chrono::high_resolution_clock::now();
//                 e.end_ns      = to_ns(start_total, now);
//                 e.duration_ns = (e.end_ns - e.start_ns);
//             }
//             event2log[client].erase(it);
//         }
//     };

//     while (1) {
//         std::vector<func_record*> frecords(num_clients, NULL);
//         size = 0;

//         // ----------------------------
//         // Pull one record per client
//         // ----------------------------
//         for (int i = 0; i < num_clients; i++) {


//             if (seen[i] == num_client_kernels[i])
//                 continue;

//             pthread_mutex_lock(client_mutexes[i]);
//             volatile int sz = client_buffers[i]->size();
//             if (sz > 0) {
//                 frecords[i] = &(client_buffers[i]->front());
//                 int cur_iter = num_client_cur_iters[i];
//                 if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {

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

//         bool canSchedule[num_clients];

//         // Close finished events -> fill end_ns/duration_ns AND UNSET MASK
//         for (int i = 0; i < num_clients; ++i) {
//             canSchedule[i] = true;
//             if (event_ids[i] >= 1) {
//                 if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
//                     auto now    = std::chrono::high_resolution_clock::now();
//                     int  ev_done = event_ids[i] - 1;
//                     auto it      = event2log[i].find(ev_done);
//                     if (it != event2log[i].end()) {
//                         Scheduler::KernelLogEntry& e = kernel_logs[it->second];
//                         if (e.end_ns == 0) {
//                             e.end_ns      = to_ns(start_total, now);
//                             e.duration_ns = (e.end_ns - e.start_ns);
//                         }
//                         event2log[i].erase(it);
//                     }
//                     // IMPORTANT: unset mask when a kernel finishes
//                     unsetmask(i);
//                 } else {
//                     canSchedule[i] = false;
//                 }
//             }
//         }

//         int num_all_clients = num_clients;
//         for (int i = 0; i < num_clients; i++) {
//             if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
//                 (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
//                 (stop_ack[i] == true)) {
//                 num_all_clients -= 1;
//             }
//         }

//         // ------------------------------------------
//         // Schedule everything available
//         // ------------------------------------------
//         for (int j = 0; j < num_clients; ++j) {
//             if (frecords[j] != NULL) {

//             if (frecords[j]->type != MALLOC_RECORD &&
//                 frecords[j]->type != MEMCPY_RECORD &&
//                 frecords[j]->type != MEMSET_RECORD &&
//                 frecords[j]->type != FREE_RECORD) {

//                 if (canSchedule[j] == false) {
//                     continue;
//                 }

//                 // Compute kernel:
//                 // 1) Decide TPCs to use via knee-tpc hint
                
//                 // op_info opj        = op_info_vector[j][seen[j]];
//                 // int     knee_tpc   = opj.opt_tpc_exclusive;
//                 // int     assigned_tpc = min(num_tpcs,knee_tpc);
//                 // if (assigned_tpc < 1) continue; // safety
//                 // setmask(assigned_tpc, j);

//                 // 2) Logging: only if NOT warmup
//                 if (!warmup) {
//                     int ev_id_before      = event_ids[j];
//                     auto now              = std::chrono::high_resolution_clock::now();
//                     int  kernel_id_before = seen[j];
//                     std::string kname =
//                         (kernel_id_before >= 0 && kernel_id_before < (int)op_info_vector[j].size())
//                         ? op_info_vector[j][kernel_id_before].name
//                         : "UNKNOWN";

//                     Scheduler::KernelLogEntry entry{
//                         /*client_id*/   j,
//                         /*iter_id  */   num_client_cur_iters[j],
//                         /*kernel_id*/   kernel_id_before,
//                         /*event_id */   ev_id_before,
//                         /*tpc_used*/    12,     // record mask size we will set
//                         /*kernel_name*/ kname,
//                         /*start_ns*/    to_ns(start_total, now),
//                         /*end_ns  */    0,
//                         /*duration*/    0
//                     };
//                     event2log[j][ev_id_before] = (int)kernel_logs.size();
//                     kernel_logs.push_back(entry);
//                 }

//                 // 4) Schedule
//                 schedule_kernel(*(frecords[j]), sched_streams[j], j,
//                                 events[j][event_ids[j]], seen, event_ids, j);
//                 pop_from_queue(client_buffers[j], client_mutexes[j], j);
//                 } else {
//                     // mem/admin kernel: schedule without logging and without mask
//                     schedule_kernel(*(frecords[j]), sched_streams[j], j,
//                                     events[j][event_ids[j]], seen, event_ids, j);
//                     pop_from_queue(client_buffers[j], client_mutexes[j], j);
//                 }
//             }
//         }

//         int finished = 0;
//         for (int i = 0; i < num_clients; i++) {

//             if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
//                 (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
//                 (stop_ack[i] == true)) {
//                 finished += 1;
//             } else if (seen[i] == num_client_kernels[i]) {
//                 // check if GPU work for this client has finished
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
//                 } else {
//                     if (event_ids[i] >= 1) {
//                         if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess)
//                             ready &= false;
//                     }
//                 }
//                 if (ready) {
//                     // IMPORTANT: unset mask here too when client's batch finishes
//                     unsetmask(i);

//                     // reset meta-structures for this client, and let it continue
//                     seen[i] = 0;
//                     if (seq)
//                         event_ids[0] = 0;
//                     event_ids[i] = 0;
//                     streams[i]   = -1;
//                     fidx[i]      = 0;
//                     request_status[i][num_client_cur_iters[i]] = true;
//                     pthread_mutex_unlock(client_mutexes[i]);
//                     num_client_cur_iters[i] += 1;
//                     locked[i] = false;
//                     client_progress[i] = 0;
//                     auto end = std::chrono::high_resolution_clock::now();
//                     float duration_ms =
//                         std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count()
//                         / 1000.0f;
//                     client_durations[i].push_back(duration_ms);
//                 }
//             }
//         }

//         if (finished == num_clients)
//             break;
//     }

//     if (!warmup) {
//         auto end_total = std::chrono::high_resolution_clock::now();
//         long long duration_ns =
//             std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
//         printf("Total loop took %lld ns\n", duration_ns);
//     }

//     if (!warmup) {
//         // Close remaining events as you already do...
//         for (int i = 0; i < num_clients; ++i) {
//             if (event_ids[i] >= 1) {
//                 int ev_last = event_ids[i] - 1;
//                 close_event_log(i, ev_last);
//             }
//         }

//         // One CSV per client, time shifted to start at 0
//         write_kernel_logs_per_client("kernel_logs", "ms_kernel_schedule_log", kernel_logs);
//     }

//     return NULL;
// }


// profile single kernel
void* Scheduler::busy_wait_ms(int num_clients,
                              int iter,
                              bool warmup,
                              int warmup_iters,
                              bool seq,
                              int depth,
                              int hp_limit,
                              int update_start) {
    DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);

    auto start_total = std::chrono::high_resolution_clock::now();

    // ────────────────────────────────────────────────────────────────────────
    // Model names per client (so we can log which model a kernel belongs to)
    // ────────────────────────────────────────────────────────────────────────
    std::vector<std::string> model_by_client(num_clients);
    for (int i = 0; i < num_clients; ++i) {
        if (i < (int)model_names.size()) model_by_client[i] = model_names[i];
        else                            model_by_client[i] = "model" + std::to_string(i);
    }

    std::vector<bool> total_client_set(num_clients, false);

    // =========================
    // Logging (ns scale; only what we need for per-kernel latency)
    // =========================
    auto to_ns = [](const std::chrono::high_resolution_clock::time_point& t0,
                    const std::chrono::high_resolution_clock::time_point& t) -> long long {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count();
    };

    std::vector<Scheduler::KernelLogEntry>               kernel_logs;
    std::vector<std::unordered_map<int, int>> event2log(num_clients); // per-client: event_id -> index in kernel_logs

    auto close_event_log = [&](int client, int ev_done) {
        auto it = event2log[client].find(ev_done);
        if (it != event2log[client].end()) {
            auto now = std::chrono::high_resolution_clock::now();
            Scheduler::KernelLogEntry& e = kernel_logs[it->second];
            if (e.end_ns == 0) {
                e.end_ns      = to_ns(start_total, now);
                e.duration_ns = e.end_ns - e.start_ns;
            }
            event2log[client].erase(it);
        }
    };

    // =========================
    // TPC masking policy
    // Start masking from the 11th iteration:
    // if num_client_cur_iters is 0-based: 11th iter => iter0 == 10
    // iter0=10 -> 1 TPC, iter0=11 -> 2 TPC, ..., iter0=33 -> 24 TPC, iter0=34 -> 1 TPC (wrap)
    // =========================
    constexpr int kTotalTpcs       = 24;
    constexpr int kMaskStartIter0  = 10;  // 0-based

    auto tpc_for_iter0 = [&](int iter0) -> int {
        return ((iter0 - kMaskStartIter0) % kTotalTpcs) + 1; // 1..24, wraps
    };

    while (1) {
        std::vector<func_record*> frecords(num_clients, NULL);

        // ----------------------------
        // Pull one record per client
        // ----------------------------
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

        bool canSchedule[num_clients];

        // ----------------------------
        // Close finished events -> fill end_ns/duration_ns and release mask
        // ----------------------------
        for (int i = 0; i < num_clients; ++i) {
            canSchedule[i] = true;
            if (event_ids[i] >= 1) {
                if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                    auto now     = std::chrono::high_resolution_clock::now();
                    int  ev_done = event_ids[i] - 1;

                    auto it = event2log[i].find(ev_done);
                    if (it != event2log[i].end()) {
                        Scheduler::KernelLogEntry& e = kernel_logs[it->second];
                        if (e.end_ns == 0) {
                            e.end_ns      = to_ns(start_total, now);
                            e.duration_ns = (e.end_ns - e.start_ns);
                        }
                        event2log[i].erase(it);
                    }

                    unsetmask(i);
                } else {
                    canSchedule[i] = false;
                }
            }
        }

        // ------------------------------------------
        // Schedule everything available
        // ------------------------------------------
        for (int j = 0; j < num_clients; ++j) {
            if (frecords[j] != NULL) {

                const bool is_mem_or_admin =
                    (frecords[j]->type == MALLOC_RECORD) ||
                    (frecords[j]->type == MEMCPY_RECORD) ||
                    (frecords[j]->type == MEMSET_RECORD) ||
                    (frecords[j]->type == FREE_RECORD);

                if (!is_mem_or_admin) {
                    if (canSchedule[j] == false) {
                        continue;
                    }

                    if (!warmup) {
                        const int iter0 = num_client_cur_iters[j];

                        // Masking starts at the 11th iteration (iter0 == 10)
                        const bool use_mask = (iter0 >= kMaskStartIter0);
                        const int  tpc_used = use_mask ? tpc_for_iter0(iter0) : kTotalTpcs;

                        int  ev_id_before      = event_ids[j];
                        int  kernel_id_before  = seen[j];
                        auto now               = std::chrono::high_resolution_clock::now();

                        std::string kname =
                            (kernel_id_before >= 0 && kernel_id_before < (int)op_info_vector[j].size())
                                ? op_info_vector[j][kernel_id_before].name
                                : "UNKNOWN";

                        // NEW: model name for this kernel (per client)
                        const std::string& mname = model_by_client[j];

                        if (use_mask) {
                            setmask(tpc_used, j); // 1..24
                        } else {
                            unsetmask(j);
                        }

                        // NOTE: Scheduler::KernelLogEntry must have a field for model name.
                        // Add it to your struct definition:
                        //   std::string model_name;
                        Scheduler::KernelLogEntry entry{
                            /*client_id*/    j,
                            /*iter_id  */    iter0,
                            /*kernel_id*/    kernel_id_before,
                            /*event_id */    ev_id_before,
                            /*tpc_used*/     tpc_used,
                            /*kernel_name*/  kname,
                            /*model_name*/   mname,                     // <----- ADDED
                            /*start_ns*/     to_ns(start_total, now),
                            /*end_ns  */     0,
                            /*duration_ns*/  0
                        };

                        event2log[j][ev_id_before] = (int)kernel_logs.size();
                        kernel_logs.push_back(std::move(entry));
                    }

                    // Schedule (compute)
                    schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                    events[j][event_ids[j]], seen, event_ids, j);
                    pop_from_queue(client_buffers[j], client_mutexes[j], j);
                }
                else {
                    // mem/admin: never masked, never logged
                    schedule_kernel(*(frecords[j]), sched_streams[j], j,
                                    events[j][event_ids[j]], seen, event_ids, j);
                    pop_from_queue(client_buffers[j], client_mutexes[j], j);
                }
            }
        }

        int finished = 0;
        for (int i = 0; i < num_clients; i++) {

            if (
                (num_client_cur_iters[i] == num_client_max_iters[i]) ||
                (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
                (stop_ack[i] == true)
            ) {
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

                    // reset meta-structures for this client, and let it continue
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
                    client_progress[i] = 0;

                    auto end = std::chrono::high_resolution_clock::now();
                    float duration_ms =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count()
                        / 1000.0f;
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

    if (!warmup) {
        // Close remaining events
        for (int i = 0; i < num_clients; ++i) {
            if (event_ids[i] >= 1) {
                int ev_last = event_ids[i] - 1;
                close_event_log(i, ev_last);
            }
        }

        // One CSV per client
        write_kernel_logs_per_client("kernel_logs", "ms_kernel_schedule_log", kernel_logs);
    }

    return NULL;
}






// test schedule for my algorithm
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {
		
// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	string filename = "/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_small_models/kernel_groups/Vnet_8_groups.csv";
//     vector<string> headers;
//     vector<Scheduler::KernelData> kernels = readCSV(filename, headers);

// 	unordered_map<string, unordered_map<int, int>> model_to_cluster_map;

// 	for (const auto& kernel : kernels) {
// 		model_to_cluster_map[kernel.Model][kernel.Kernel_ID] = kernel.Cluster;
// 	}

// 	std::vector<std::string> model_names = {"Vnet_8", "Vnet_8", "Vnet_8", "Vnet_8"};
// 	// std::vector<std::string> model_names = {"Rnet_8", "Rnet_8"};
// 	unordered_map<int, unordered_map<int, int>> kernel_cluster_map;

//     for (int i = 0; i < model_names.size(); i++) {
//         kernel_cluster_map[i] = model_to_cluster_map[model_names[i]];
//     }

// 	int num_groups = 0;
// 	for (const auto& kernel : kernels) {
// 		if (kernel.Cluster > num_groups) {
// 			num_groups = kernel.Cluster;  
// 		}
// 	}
// 	num_groups += 1;

// 	string filename_cm = "/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_small_models/contention_matrix/contention_matrix_Rnet_8.csv";
//     map<pair<int, int>, double> contention_map = readContentionMatrix(filename_cm);

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();
// 	auto start_micro = std::chrono::duration_cast<std::chrono::microseconds>(start_total.time_since_epoch());
// 	// std::cout << "Start time: " << start_micro.count() << " μs" << std::endl;
	
// 	std::chrono::time_point<std::chrono::system_clock> start_profile;

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

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


//     int num_kernels[num_clients];

//     // Calculate the number of kernels for each client
//     for (int i = 0; i < num_clients; i++) {
//         num_kernels[i] = (num_client_max_iters[i] - 10) * num_client_kernels[i];
//         printf("Client %d: num of total kernels = %d\n", i, num_kernels[i]);
//     }


// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

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
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}

// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}
		
// 		if(can_schedule){

// 			bool canSchedule[num_clients];
// 			for (int i = 0; i < num_clients; ++i) {
// 				canSchedule[i] = true;
// 				if (event_ids[i] >= 1) {
// 					if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 						// unsetmask(i);
// 						unsetmask_m(i);
// 					}
// 					else{
// 						canSchedule[i] = false; 
// 					}
// 				}
// 			}
// 			std::set<int> critical_clients_set;
// 			for (int client_id : ready_client) {
// 				op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 				if (op_info_cur.kernel_type == "k1" || op_info_cur.kernel_type == "k2") {
// 					critical_clients_set.insert(client_id);
// 				}
// 			}
// 			// if (critical_clients_set.size() > 1) {
// 			// 	auto it = critical_clients_set.begin(); 
// 			// 	int first_client = *it; 
// 			// 	int second_client = -1;
// 			// 	double max_contention = -1;
			
// 			// 	// Compare every pair of clients in the critical set
// 			// 	for (auto it1 = critical_clients_set.begin(); it1 != critical_clients_set.end(); ++it1) {
// 			// 		int client1 = *it1;
// 			// 		for (auto it2 = std::next(it1); it2 != critical_clients_set.end(); ++it2) {
// 			// 			int client2 = *it2;
// 			// 			int client_1_group = kernel_cluster_map[client1][seen[client1]];
// 			// 			int client_2_group = kernel_cluster_map[client2][seen[client2]];
// 			// 			pair<int, int> key = {client_1_group, client_2_group};
// 			// 			double contention = contention_map[key];
// 			// 			if (contention > max_contention) {
// 			// 				max_contention = contention;
// 			// 				first_client = client1;
// 			// 				second_client = client2;
// 			// 			}
// 			// 		}
// 			// 	}
			
// 			// 	// If two clients with the highest contention are found, keep them in the set
// 			// 	if (second_client != -1) {
// 			// 		for (auto it = critical_clients_set.begin(); it != critical_clients_set.end(); ) {
// 			// 			if (*it != first_client && *it != second_client) {
// 			// 				it = critical_clients_set.erase(it);
// 			// 			} else {
// 			// 				++it;
// 			// 			}
// 			// 		}
// 			// 	}
// 			// }

// 			// if (critical_clients_set.size() > 1) {
// 			// 	auto it = critical_clients_set.begin(); 
// 			// 	int first_client = *it; 
// 			// 	int best_client = -1;
// 			// 	double min_contention = -1;
		
// 			// 	for (auto it2 = std::next(it); it2 != critical_clients_set.end(); ++it2) {
// 			// 		int other_client = *it2;
// 			// 		int client_1_group = kernel_cluster_map[first_client][seen[first_client]];
// 			// 		int client_2_group = kernel_cluster_map[other_client][seen[other_client]];
// 			// 		pair<int, int> key = {client_1_group, client_2_group};
// 			// 		double contention = contention_map[key];
// 			// 		if (contention > min_contention && contention > 0.8) {
// 			// 			min_contention = contention;
// 			// 			best_client = other_client;
// 			// 		}
// 			// 	}
			
// 			// 	if (best_client != -1) {
// 			// 		for (auto it = critical_clients_set.begin(); it != critical_clients_set.end(); ) {
// 			// 			if (*it != first_client && *it != best_client) {
// 			// 				it = critical_clients_set.erase(it);
// 			// 			} else {
// 			// 				++it;
// 			// 			}
// 			// 		}
// 			// 	}
// 			// }

// 			for (int client_id : ready_client) {

// 				if (frecords[client_id] != NULL) {

// 					op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 					if(critical_clients_set.count(client_id) && canSchedule[client_id]){

// 						if(critical_clients_set.size() > 1){
// 							int tpc_usage = 24;
// 							if(num_tpcs >= tpc_usage){
// 								setmask_m(24, client_id, 2);
// 								schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 								pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 							}
// 						}
// 						else{
// 							int tpc_usage = op_info_cur.knee_tpc;
// 							if(num_tpcs >= tpc_usage){
// 								if(num_all_clients == 1){
// 									tpc_usage = 24;
// 								}
// 								setmask_m(tpc_usage, client_id, 1);
// 								schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 								pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 							}
// 						}
// 					}
					
// 					if(op_info_cur.kernel_type == "k3"){
// 						int tpc_usage = op_info_cur.knee_tpc;
// 						setmask_O(tpc_usage, client_id);
// 						schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 						pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);					
// 					}
// 				}
//                 // schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
//                 // pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);		
// 			}
// 		}


// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
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
// 					// printf("finish %d\n",i)
// 					unsetmask_m(i);
// 					// unsetmask(i);
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
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
//     	auto duration= std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %ld nanoseconds\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
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

static inline int client_active_tpcs(int client_id) {
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
    int existing_tpcs)
{
  ProfileCheck out{};
  if (!cur) return out;

  const int n_excl_opt  = std::max(1, cur->opt_tpc_exclusive);
  const int n_coloc_opt = std::max(1, cur->opt_tpc_colocated);

  out.TPC_IND_OPT = n_excl_opt;
  out.TPC_COR_OPT = n_coloc_opt;

  // running summary (other clients)
  int used_tpcs_other = 0;
  for (const auto* r : running) {
    if (r) used_tpcs_other += std::max(0, r->tpc_used);
  }

//   printf("[CHK] cur_op id=%d name=%s cluster=%d | IND_OPT=%d COR_OPT=%d | "
//          "running(other)=%zu used_other=%d num_tpcs_global=%d existing_tpcs=%d\n",
//          cur->id, cur->name.c_str(), cur->cluster,
//          n_excl_opt, n_coloc_opt,
//          running.size(), used_tpcs_other, num_tpcs, existing_tpcs);

  // -------- TIME_IND_OPT at exclusive knee --------
  const float t_ind = excl_at_n(cur, n_excl_opt);
  out.TIME_IND_OPT = t_ind;
//   if (std::isfinite(t_ind))
    // printf("[CHK] TIME_IND_OPT = excl_us_by_n[%d] = %.3f us\n", n_excl_opt, t_ind);
//   else
    // printf("[CHK] TIME_IND_OPT = NaN (missing exclusive at n=%d)\n", n_excl_opt);

  // -------- UT ratio at co-run knee (n = TPC_COR_OPT) --------
  const double base_ut = ut_ratio(cur->cluster, n_coloc_opt);
//   printf("[CHK] UT ratio (UT=%d, n=%d) = %.4f\n",
//          cur->cluster, n_coloc_opt, base_ut);

  // -------- worst-case pair multiplier vs running at knee --------
  double worst_ratio = base_ut;
  for (const auto* r : running) {
    if (!r) continue;
    const double r_pair = pair_ratio(cur->cluster, r->cluster, n_coloc_opt);
    const double rco    = std::max(r_pair, base_ut);  // fallback to UT if pair weak/missing
    // printf("[CHK] vs running(cluster=%d) @n=%d: pair=%.4f, ut=%.4f -> use=%.4f\n",
    //        r->cluster, n_coloc_opt, r_pair, base_ut, rco);
    if (rco > worst_ratio) worst_ratio = rco;
  }
//   printf("[CHK] worst_ratio at COR_OPT n=%d = %.4f\n",
//          n_coloc_opt, worst_ratio);

  // -------- TIME_COR_OPT = excl(n_coloc_opt) * worst_ratio --------
  const float t_excl_cor = excl_at_n(cur, n_coloc_opt);
//   if (std::isfinite(t_excl_cor))
//     printf("[CHK] excl_us_by_n[%d] = %.3f us\n", n_coloc_opt, t_excl_cor);
//   else
//     printf("[CHK] excl_us_by_n[%d] = NaN\n", n_coloc_opt);

  out.TIME_COR_OPT = (std::isfinite(t_excl_cor)
                      ? float(t_excl_cor * worst_ratio)
                      : NAN);
//   if (std::isfinite(out.TIME_COR_OPT))
//     printf("[CHK] TIME_COR_OPT = %.3f * %.4f = %.3f us\n",
//            t_excl_cor, worst_ratio, out.TIME_COR_OPT);
//   else
//     printf("[CHK] TIME_COR_OPT = NaN\n");

  // ================== FIXED AVAILABLE-PART BELOW ==================

  // -------- pick n_avail from num_tpcs + existing_tpcs --------
  int raw_avail = num_tpcs + std::max(0, existing_tpcs);
  int n_avail   = std::max(1, std::min(n_coloc_opt, raw_avail));

//   printf("[CHK] N_AVAIL for client = num_tpcs(%d) + existing_tpcs(%d) = %d "
//          " → n_avail(clamped to COR_OPT=%d) = %d\n",
//          num_tpcs, existing_tpcs, raw_avail, n_coloc_opt, n_avail);

  // UT ratio at n_avail
  const double base_ut_av = ut_ratio(cur->cluster, n_avail);
//   printf("[CHK] UT ratio_avail (UT=%d, n=%d) = %.4f\n",
//          cur->cluster, n_avail, base_ut_av);

  // worst-case pair multiplier vs running at n_avail
  double worst_ratio_av = base_ut_av;
  for (const auto* r : running) {
    if (!r) continue;
    const double r_pair_av = pair_ratio(cur->cluster, r->cluster, n_avail);
    const double rco_av    = std::max(r_pair_av, base_ut_av);
    // printf("[CHK] vs running(cluster=%d) @n=%d: pair_av=%.4f, ut_av=%.4f -> use_av=%.4f\n",
    //        r->cluster, n_avail, r_pair_av, base_ut_av, rco_av);
    if (rco_av > worst_ratio_av) worst_ratio_av = rco_av;
  }
//   printf("[CHK] worst_ratio_av at n=%d = %.4f\n", n_avail, worst_ratio_av);

  // -------- TIME_COR_AVAIL = excl(n_avail) * worst_ratio_av --------
  const float t_excl_av = excl_at_n(cur, n_avail);
//   if (std::isfinite(t_excl_av))
//     printf("[CHK] excl_us_by_n[%d]=%.3f us (for TIME_COR_AVAIL)\n",
//            n_avail, t_excl_av);
//   else
//     printf("[CHK] excl_us_by_n[%d] = NaN\n", n_avail);

  out.TIME_COR_AVAIL = (std::isfinite(t_excl_av)
                        ? float(t_excl_av * worst_ratio_av)
                        : NAN);
//   if (std::isfinite(out.TIME_COR_AVAIL))
//     printf("[CHK] TIME_COR_AVAIL = %.3f * %.4f = %.3f us\n",
//            t_excl_av, worst_ratio_av, out.TIME_COR_AVAIL);
//   else
//     printf("[CHK] TIME_COR_AVAIL = NaN\n");

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


// Update global g_running_ops based on predicted finish times (est_finish_ns),
// shrink/unset masks accordingly, and rebuild out_running (excluding one client).
// Additionally, track per-client last finished kernel id for window barriers.
static inline void update_and_build_running_ops(
    uint64_t now_ns,
    int exclude_client_id,
    std::vector<const op_info*>& out_running,
    std::vector<int>& last_finished_kernel_id)
{
    // using RunningOp = Scheduler::RunningOp;
    auto& vec = g_running_ops;

    std::vector<RunningOp> new_vec;
    new_vec.reserve(vec.size());

    for (auto &rk : vec) {
        bool finished = false;

        if (!rk.op) {
            finished = true;
        } else if (!rk.op->is_running) {
            finished = true;
        } else if (rk.op->est_finish_ns <= now_ns) {
            // finished according to predicted finish time
            finished = true;
        }

        if (finished) {
            if (rk.op) {
                rk.op->is_running = false;
            }

            // Update last finished kernel id for this client (for sync window)
            if (rk.client_id >= 0 &&
                rk.client_id < (int)last_finished_kernel_id.size())
            {
                if (rk.kernel_id > last_finished_kernel_id[rk.client_id]) {
                    last_finished_kernel_id[rk.client_id] = rk.kernel_id;
                }
            }

            // compute how many TPCs are still needed by other running kernels
            // of this *same* client
            int needed_tpcs = 0;
            for (const auto &other : vec) {
                if (&other == &rk) continue;               // skip this finished entry
                if (other.client_id != rk.client_id) continue;
                if (!other.op || !other.op->is_running) continue;
                if (other.op->tpc_used > needed_tpcs) {
                    needed_tpcs = other.op->tpc_used;
                }
            }

            if (needed_tpcs == 0) {
                // no other kernels from this client; free all TPCs
                unsetmask(rk.client_id);
            } else {
                // release only extra bits
                shrink_mask_for_client(rk.client_id, needed_tpcs);
            }

            // DO NOT push rk into new_vec → we drop it
        } else {
            new_vec.push_back(rk);
        }
    }

    vec.swap(new_vec);

    // rebuild "running ops" list for profile_check, skipping exclude_client_id
    out_running.clear();
    out_running.reserve(vec.size());
    for (const auto &rk : vec) {
        if (rk.op && rk.op->is_running && rk.client_id != exclude_client_id) {
            out_running.push_back(rk.op);
        }
    }

    // std::printf("[RUNOP] after update: total_running=%zu (excluding client %d)\n",
    //             vec.size(), exclude_client_id);
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





// void* Scheduler::busy_wait_ficks(int num_clients, int iter, bool warmup, int warmup_iters,
//                                     bool seq, int depth, int hp_limit, int update_start)
// {
//     DEBUG_PRINT("Entered busy_wait_profile (masked scheduler)! Num clients is %d\n", num_clients);

//     int start0 = 0;
//     int start1 = 0;

//     auto start_total = std::chrono::high_resolution_clock::now();

//     std::vector<bool> total_client_set(num_clients, false);
//     std::vector<int>  profiles(num_clients, -1);
//     std::vector<int>  cur_sms(num_clients, -1);

//     int hp_client = 1;
//     int lp_client = 0;

//     bool large_found = false;
//     long sum  = 0;   // sum of durations of ongoing BE kernels
//     long size = 0;   // sum of sizes of in-the-queues BE kernels
//     int  start = -1;

//     // BS - works only for 2 clients for now
//     int   low_sms        = 0;
//     int   high_sms       = max_sms_clients[0]; // 0 is the lp client
//     int   sm_threshold   = max_sms_clients[0] / 2;
//     float hp_iter_duration = 0.0f; // 1 is the hp client
//     float hp_limit_float   = (float)hp_limit;

//     auto to_ns = [](const std::chrono::high_resolution_clock::time_point& t0,
//                     const std::chrono::high_resolution_clock::time_point& t) -> long long {
//         return std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count();
//     };

//     // Helper: pick "knee TPC" from op_info; clamp [1, num_tpcs]
//     auto pick_knee_tpc = [&](const op_info& op) -> int {
//         int knee = op.knee_tpc;  // <- make sure this exists in op_info
//         if (knee < 1) knee = 1;
//         if (knee > num_tpcs) knee = num_tpcs;
//         return knee;
//     };

//     std::vector<Scheduler::KernelLogEntry>               kernel_logs;
//     std::vector<std::unordered_map<int, int>> event2log(num_clients); // per-client: event_id -> index in kernel_logs


//     auto close_event_log = [&](int client, int ev_done) {
//         auto it = event2log[client].find(ev_done);
//         if (it != event2log[client].end()) {
//             Scheduler::KernelLogEntry& e = kernel_logs[it->second];
//             if (e.end_ns == 0) {
//                 auto now = std::chrono::high_resolution_clock::now();
//                 e.end_ns      = to_ns(start_total, now);
//                 e.duration_ns = (e.end_ns - e.start_ns);
//             }
//             event2log[client].erase(it);
//         }
//     };

//     while (1) {
//         std::vector<func_record*> frecords(num_clients, NULL);
//         size = 0;

//         // ----------------------------
//         // Pull one record per client
//         // ----------------------------
//         for (int i = 0; i < num_clients; i++) {


//             if (seen[i] == num_client_kernels[i])
//                 continue;

//             pthread_mutex_lock(client_mutexes[i]);
//             volatile int sz = client_buffers[i]->size();
//             if (sz > 0) {
//                 frecords[i] = &(client_buffers[i]->front());
//                 int cur_iter = num_client_cur_iters[i];
//                 if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {

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

//         bool canSchedule[num_clients];

//         // Close finished events -> fill end_ns/duration_ns AND UNSET MASK
//         for (int i = 0; i < num_clients; ++i) {
//             canSchedule[i] = true;
//             if (event_ids[i] >= 1) {
//                 if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
//                     auto now    = std::chrono::high_resolution_clock::now();
//                     int  ev_done = event_ids[i] - 1;
//                     auto it      = event2log[i].find(ev_done);
//                     if (it != event2log[i].end()) {
//                         Scheduler::KernelLogEntry& e = kernel_logs[it->second];
//                         if (e.end_ns == 0) {
//                             e.end_ns      = to_ns(start_total, now);
//                             e.duration_ns = (e.end_ns - e.start_ns);
//                         }
//                         event2log[i].erase(it);
//                     }
//                     // IMPORTANT: unset mask when a kernel finishes
//                     unsetmask(i);
//                 } else {
//                     canSchedule[i] = false;
//                 }
//             }
//         }

//         int num_all_clients = num_clients;
//         for (int i = 0; i < num_clients; i++) {
//             if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
//                 (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
//                 (stop_ack[i] == true)) {
//                 num_all_clients -= 1;
//             }
//         }

//         // ------------------------------------------
//         // Schedule everything available
//         // ------------------------------------------
//         for (int j = 0; j < num_clients; ++j) {
//             if (frecords[j] != NULL) {

//             if (frecords[j]->type != MALLOC_RECORD &&
//                 frecords[j]->type != MEMCPY_RECORD &&
//                 frecords[j]->type != MEMSET_RECORD &&
//                 frecords[j]->type != FREE_RECORD) {

//                 if (canSchedule[j] == false) {
//                     continue;
//                 }

//                 // Compute kernel:
//                 // 1) Decide TPCs to use via knee-tpc hint
                
//                 op_info opj        = op_info_vector[j][seen[j]];
//                 int     knee_tpc   = opj.opt_tpc_exclusive;
//                 int     assigned_tpc = min(num_tpcs,knee_tpc);
//                 if (assigned_tpc < 1) continue; // safety
//                 setmask(assigned_tpc, j);

//                 // 2) Logging: only if NOT warmup
//                 if (!warmup) {
//                     int ev_id_before      = event_ids[j];
//                     auto now              = std::chrono::high_resolution_clock::now();
//                     int  kernel_id_before = seen[j];
//                     std::string kname =
//                         (kernel_id_before >= 0 && kernel_id_before < (int)op_info_vector[j].size())
//                         ? op_info_vector[j][kernel_id_before].name
//                         : "UNKNOWN";

//                     Scheduler::KernelLogEntry entry{
//                         /*client_id*/   j,
//                         /*iter_id  */   num_client_cur_iters[j],
//                         /*kernel_id*/   kernel_id_before,
//                         /*event_id */   ev_id_before,
//                         /*tpc_used*/    assigned_tpc,     // record mask size we will set
//                         /*kernel_name*/ kname,
//                         /*start_ns*/    to_ns(start_total, now),
//                         /*end_ns  */    0,
//                         /*duration*/    0
//                     };
//                     event2log[j][ev_id_before] = (int)kernel_logs.size();
//                     kernel_logs.push_back(entry);
//                 }

//                 // 4) Schedule
//                 schedule_kernel(*(frecords[j]), sched_streams[j], j,
//                                 events[j][event_ids[j]], seen, event_ids, j);
//                 pop_from_queue(client_buffers[j], client_mutexes[j], j);
//                 } else {
//                     // mem/admin kernel: schedule without logging and without mask
//                     schedule_kernel(*(frecords[j]), sched_streams[j], j,
//                                     events[j][event_ids[j]], seen, event_ids, j);
//                     pop_from_queue(client_buffers[j], client_mutexes[j], j);
//                 }
//             }
//         }

//         int finished = 0;
//         for (int i = 0; i < num_clients; i++) {

//             if ((num_client_cur_iters[i] == num_client_max_iters[i]) ||
//                 (warmup && (num_client_cur_iters[i] == warmup_iters)) ||
//                 (stop_ack[i] == true)) {
//                 finished += 1;
//             } else if (seen[i] == num_client_kernels[i]) {
//                 // check if GPU work for this client has finished
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
//                 } else {
//                     if (event_ids[i] >= 1) {
//                         if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess)
//                             ready &= false;
//                     }
//                 }
//                 if (ready) {
//                     // IMPORTANT: unset mask here too when client's batch finishes
//                     unsetmask(i);

//                     // reset meta-structures for this client, and let it continue
//                     seen[i] = 0;
//                     if (seq)
//                         event_ids[0] = 0;
//                     event_ids[i] = 0;
//                     streams[i]   = -1;
//                     fidx[i]      = 0;
//                     request_status[i][num_client_cur_iters[i]] = true;
//                     pthread_mutex_unlock(client_mutexes[i]);
//                     num_client_cur_iters[i] += 1;
//                     locked[i] = false;
//                     client_progress[i] = 0;
//                     auto end = std::chrono::high_resolution_clock::now();
//                     float duration_ms =
//                         std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count()
//                         / 1000.0f;
//                     client_durations[i].push_back(duration_ms);
//                 }
//             }
//         }

//         if (finished == num_clients)
//             break;
//     }

//     if (!warmup) {
//         auto end_total = std::chrono::high_resolution_clock::now();
//         long long duration_ns =
//             std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
//         printf("Total loop took %lld ns\n", duration_ns);
//     }

//     if (!warmup) {
//         // Close remaining events as you already do...
//         for (int i = 0; i < num_clients; ++i) {
//             if (event_ids[i] >= 1) {
//                 int ev_last = event_ids[i] - 1;
//                 close_event_log(i, ev_last);
//             }
//         }

//         // One CSV per client, time shifted to start at 0
//         write_kernel_logs_per_client("kernel_logs", "krisp_kernel_schedule_log", kernel_logs);
//     }

//     return NULL;
// }


void* Scheduler::busy_wait_ficks(int num_clients, int iter, bool warmup, int warmup_iters,
                                 bool seq, int depth, int hp_limit, int update_start)
{

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
        // Special case: mobilenet_v2 → mobilenetv2 (matches your CSV filenames)
        if (s == "mobilenet_v2") return std::string("mobilenetv2");
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

    schedule_client[0] = pair.first.tpc;
    schedule_client[1] = pair.second.tpc;

    // ────────────────────────────────────────────────────────────────────────
    // 3) Timers / bookkeeping
    // ────────────────────────────────────────────────────────────────────────
    auto start_total = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> start_profile;

    std::vector<bool> total_client_set(num_clients, false);

    int  hp_client = 1, lp_client = 0;
    int  low_sms   = 0;
    int  high_sms  = max_sms_clients[0];
    int  sm_threshold = max_sms_clients[0] / 2;
    float hp_iter_duration = 0.0f;
    float hp_limit_float   = (float)hp_limit;

    bool finish_profile[2] = {false, false};
    bool launched_c[2]     = {false, false};  // window kernels enqueued
    bool finished_c[2]     = {false, false};  // window kernels completed
    bool printed_pair      = false;           // print-once guard
    bool store_profile     = false;           // record once per pair
    std::chrono::time_point<std::chrono::high_resolution_clock> finish_ts[2];

    // ────────────────────────────────────────────────────────────────────────
    // 4) In-memory record struct (we'll write immediately, no big vector)
    // ────────────────────────────────────────────────────────────────────────
    struct PairRecord {
        int         iteration;        // iteration index when measured
        int         kid0, kid1;       // window_start ids for the pair
        int         tpc0, tpc1;       // masks used for the pair
        long long   client0_wall_ns;  // client-0 wall time
        std::string model0;           // logical model name for client 0
        std::string model1;           // logical model name for client 1
    };

    // ────────────────────────────────────────────────────────────────────────
    // 5) Main loop
    // ────────────────────────────────────────────────────────────────────────
    while (1) {
        // Track if a client has scheduled anything in THIS while-iteration
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

        // (5.2) Warmup gating
        bool can_schedule = true;
        for (int j = 0; j < num_clients; ++j) {
            if (num_client_cur_iters[j] < 10) {  // 10 warmup iters
                can_schedule = false;
            }
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
        if (can_schedule) {
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
                        launched_c[0] = launched_c[1] = false;
                        store_profile = false;

                        for (int client_id : ready_client) {
                            if (frecords[client_id] != NULL &&
                                seen[client_id] < window_end[client_id])
                            {
                                if (client_id == 0) {
                                    start_profile = std::chrono::high_resolution_clock::now();
                                }
                                setmask(schedule_client[client_id], client_id);
                                scheduled_in_tick[client_id] = true;

                                schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id,
                                                events[client_id][event_ids[client_id]], seen, event_ids, client_id);
                                pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
                            }
                        }
                    }
                } else if (seen[0] == window_start[0] && seen[1] == window_start[1] &&
                           any_scheduled_this_tick)
                {
                    // skip window-enqueue this tick
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

        // (5.8) Record client-0 wall-clock once it finishes (first time)
        if (!printed_pair && finished_c[0] && !store_profile) {
            auto end0 = finish_ts[0];
            auto wall_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - start_profile).count();

            PairRecord rec;
            rec.iteration       = num_client_cur_iters[0]; // both equal here
            rec.kid0            = window_start[0];
            rec.kid1            = window_start[1];
            rec.tpc0            = schedule_client[0];
            rec.tpc1            = schedule_client[1];
            rec.client0_wall_ns = wall_ns;
            rec.model0          = model0;   // logical names (e.g., "mobilenet_v2")
            rec.model1          = model1;

            printf("[REC] iter=%d kid0=%d kid1=%d tpc0=%d tpc1=%d wall_ns=%lld "
                   "model0=%s model1=%s\n",
                   rec.iteration,
                   rec.kid0,
                   rec.kid1,
                   rec.tpc0,
                   rec.tpc1,
                   rec.client0_wall_ns,
                   rec.model0.c_str(),
                   rec.model1.c_str());

            // ---- Append directly to CSV here ----
            try {
                bool need_header = !fs::exists(out_path);

                std::ofstream csv(out_path, std::ios::app);
                if (!csv.is_open()) {
                    std::cerr << "[ERROR] Could not open " << out_path
                              << " for writing." << std::endl;
                } else {
                    if (need_header) {
                        csv << "iteration,model0,model1,kernel0_id,kernel1_id,"
                               "tpc0,tpc1,client0_wall_ns\n";
                    }
                    csv << rec.iteration       << ","
                        << rec.model0          << ","
                        << rec.model1          << ","
                        << rec.kid0            << ","
                        << rec.kid1            << ","
                        << rec.tpc0            << ","
                        << rec.tpc1            << ","
                        << rec.client0_wall_ns << "\n";
                }
            } catch (const std::exception &e) {
                std::cerr << "[ERROR] Exception while writing CSV " << out_path
                          << ": " << e.what() << std::endl;
            }

            store_profile = true;
        }

        // (5.9) When both finished, mark window finished and advance plan
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
                schedule_client[0] = next_pair.first.tpc;
                schedule_client[1] = next_pair.second.tpc;
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
                // Check if GPU work for this client has finished
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
                        if (can_schedule) {
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
                    // reset per-client meta for next iteration
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

                    // iteration duration (kept)
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
        auto duration_nano =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
        printf("Total loop took %ld nanoseconds\n", duration_nano);
    }

    // NOTE: no final CSV dump here — everything is written incrementally.

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
                // 7: knee_tpc
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
                // info.is_on_list = std::stoi(v[8]);
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

        const char* RATIO_ROOT = "/home/zixi/orion_bu/benchmarking/model_kernels/ficks/profiling/";
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