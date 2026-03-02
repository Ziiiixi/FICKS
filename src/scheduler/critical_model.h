// ===== critical_model.h (core patch) =====
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class CriticalityModel {
public:
    struct Config {
        // smoothing
        float alpha;

        // feature bins
        int seq_bins;
        int history_n;
        int history_bins;

        // cluster hierarchy weights
        float w_c1, w_c2, w_c3, w_c4, w_c5;

        // latency hierarchy weights
        float w_l1, w_l2, w_l3, w_l4, w_l5;

        // critical threshold tuning range
        float tune_thr_lo, tune_thr_hi, tune_thr_step;

        // numerical floor for percentage metrics
        float mape_floor_us;

        Config()
            : alpha(1.5f),
              seq_bins(256),
              history_n(20),
              history_bins(6),
              w_c1(3.0f), w_c2(2.4f), w_c3(1.8f), w_c4(1.2f), w_c5(0.8f),
              w_l1(3.0f), w_l2(2.2f), w_l3(1.6f), w_l4(1.1f), w_l5(0.7f),
              tune_thr_lo(0.05f), tune_thr_hi(0.95f), tune_thr_step(0.01f),
              mape_floor_us(10.0f) {}
    };

    struct CritEval { long long correct=0, total=0; };
    struct ClusterEval { long long correct=0, total=0, unknown_pred=0; };
    struct LatEval {
        long long points=0;
        double abs_sum=0.0, ape_sum=0.0, sape_sum=0.0;
    };

    CriticalityModel() : cfg_(Config()) {}
    explicit CriticalityModel(const Config& cfg) : cfg_(cfg) {}

    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        crit_cnt_.clear();
        crit_tot_.clear();
        clu_cnt_.clear();
        clu_tot_.clear();
        lat_log_.clear();
        cluster_ids_.clear();
    }

    void set_config(const Config& cfg) {
        std::lock_guard<std::mutex> lk(mu_);
        cfg_ = cfg;
    }

    Config get_config() const {
        std::lock_guard<std::mutex> lk(mu_);
        return cfg_;
    }

    // ---------- feature helpers ----------
    int seq_bin(int seen_k, int total_k) const {
        int t = std::max(1, total_k);
        int s = std::max(0, seen_k);
        int b = (s * std::max(1, cfg_.seq_bins)) / t;
        if (b >= cfg_.seq_bins) b = cfg_.seq_bins - 1;
        if (b < 0) b = 0;
        return b;
    }

    int hist_bin_from_ratio(float r) const {
        float x = std::max(0.0f, std::min(1.0f, r));
        int hb = (int)std::floor(x * cfg_.history_bins);
        if (hb >= cfg_.history_bins) hb = cfg_.history_bins - 1;
        if (hb < 0) hb = 0;
        return hb;
    }

    // ---------- training ----------
    void observe_one(const std::string& model,
                     const std::string& exact_name,
                     int seen_k, int total_k,
                     int hist_bin,
                     int gt_is_critical,          // 1 critical, 0 non critical
                     int gt_cluster,
                     const std::vector<float>& gt_excl_us_by_n) {
        const int sb = seq_bin(seen_k, total_k);

        std::lock_guard<std::mutex> lk(mu_);

        // critical
        add_crit(K1(model, exact_name, sb, hist_bin), gt_is_critical);
        add_crit(K2(model, exact_name, sb), gt_is_critical);
        add_crit(K3(model, exact_name), gt_is_critical);
        add_crit(K4(model, sb), gt_is_critical);
        add_crit(K5(model), gt_is_critical);

        // cluster
        add_clu(K1(model, exact_name, sb, hist_bin), gt_cluster);
        add_clu(K2(model, exact_name, sb), gt_cluster);
        add_clu(K3(model, exact_name), gt_cluster);
        add_clu(K4(model, sb), gt_cluster);
        add_clu(K5(model), gt_cluster);
        cluster_ids_.insert(gt_cluster);

        // latency (log space, per tpc)
        for (int t = 1; t <= (int)gt_excl_us_by_n.size(); ++t) {
            float y = gt_excl_us_by_n[t - 1];
            if (!(std::isfinite(y) && y > 0.0f)) continue;
            double ly = std::log((double)y);

            add_lat(K1L(model, exact_name, sb, hist_bin, t), ly);
            add_lat(K2L(model, exact_name, sb, t), ly);
            add_lat(K3L(model, exact_name, t), ly);
            add_lat(K4L(model, sb, t), ly);
            add_lat(K5L(model, t), ly);
        }
    }

    // ---------- inference ----------
    float predict_critical_prob(const std::string& model,
                                const std::string& exact_name,
                                int seen_k, int total_k,
                                int hist_bin) const {
        const int sb = seq_bin(seen_k, total_k);
        std::lock_guard<std::mutex> lk(mu_);

        double num = 0.0, den = 0.0;
        blend_crit(K1(model, exact_name, sb, hist_bin), 3.0, num, den);
        blend_crit(K2(model, exact_name, sb),          2.4, num, den);
        blend_crit(K3(model, exact_name),              1.8, num, den);
        blend_crit(K4(model, sb),                      1.2, num, den);
        blend_crit(K5(model),                          0.8, num, den);

        if (den <= 0.0) return 0.5f;
        float p = (float)(num / den);
        if (p < 0.01f) p = 0.01f;
        if (p > 0.99f) p = 0.99f;
        return p;
    }

    int predict_cluster(const std::string& model,
                        const std::string& exact_name,
                        int seen_k, int total_k,
                        int hist_bin) const {
        const int sb = seq_bin(seen_k, total_k);
        std::lock_guard<std::mutex> lk(mu_);

        if (cluster_ids_.empty()) return -1;
        const double a = std::max(1e-6, (double)cfg_.alpha);
        const int C = (int)cluster_ids_.size();

        double best = -std::numeric_limits<double>::infinity();
        int best_c = -1;

        for (std::unordered_set<int>::const_iterator itc = cluster_ids_.begin();
             itc != cluster_ids_.end(); ++itc) {
            int c = *itc;
            double s = 0.0;

            s += cfg_.w_c1 * clu_logprob(K1(model, exact_name, sb, hist_bin), c, a, C);
            s += cfg_.w_c2 * clu_logprob(K2(model, exact_name, sb),          c, a, C);
            s += cfg_.w_c3 * clu_logprob(K3(model, exact_name),              c, a, C);
            s += cfg_.w_c4 * clu_logprob(K4(model, sb),                      c, a, C);
            s += cfg_.w_c5 * clu_logprob(K5(model),                          c, a, C);

            if (s > best) { best = s; best_c = c; }
        }
        return best_c;
    }

    float predict_latency_at_tpc(const std::string& model,
                                 const std::string& exact_name,
                                 int seen_k, int total_k,
                                 int hist_bin,
                                 int tpc) const {
        const int sb = seq_bin(seen_k, total_k);
        std::lock_guard<std::mutex> lk(mu_);

        double num = 0.0, den = 0.0;
        blend_lat(K1L(model, exact_name, sb, hist_bin, tpc), cfg_.w_l1, num, den);
        blend_lat(K2L(model, exact_name, sb, tpc),           cfg_.w_l2, num, den);
        blend_lat(K3L(model, exact_name, tpc),               cfg_.w_l3, num, den);
        blend_lat(K4L(model, sb, tpc),                       cfg_.w_l4, num, den);
        blend_lat(K5L(model, tpc),                           cfg_.w_l5, num, den);

        if (den <= 0.0) return std::numeric_limits<float>::quiet_NaN();
        return (float)std::exp(num / den);
    }

private:
    struct CritStat { uint64_t n=0, n1=0; };
    struct CluStat {
        uint64_t total=0;
        std::unordered_map<int, uint64_t> cnt;
    };
    struct LatStat { uint64_t n=0; double sum_log=0.0; };

    static std::string I2S(int x) { return std::to_string(x); }

    static std::string K1(const std::string& m,const std::string& n,int sb,int hb){
        return "C1|" + m + "|" + n + "|s" + I2S(sb) + "|h" + I2S(hb);
    }
    static std::string K2(const std::string& m,const std::string& n,int sb){
        return "C2|" + m + "|" + n + "|s" + I2S(sb);
    }
    static std::string K3(const std::string& m,const std::string& n){
        return "C3|" + m + "|" + n;
    }
    static std::string K4(const std::string& m,int sb){
        return "C4|" + m + "|s" + I2S(sb);
    }
    static std::string K5(const std::string& m){
        return "C5|" + m;
    }

    static std::string K1L(const std::string& m,const std::string& n,int sb,int hb,int t){
        return "L1|" + m + "|" + n + "|s" + I2S(sb) + "|h" + I2S(hb) + "|t" + I2S(t);
    }
    static std::string K2L(const std::string& m,const std::string& n,int sb,int t){
        return "L2|" + m + "|" + n + "|s" + I2S(sb) + "|t" + I2S(t);
    }
    static std::string K3L(const std::string& m,const std::string& n,int t){
        return "L3|" + m + "|" + n + "|t" + I2S(t);
    }
    static std::string K4L(const std::string& m,int sb,int t){
        return "L4|" + m + "|s" + I2S(sb) + "|t" + I2S(t);
    }
    static std::string K5L(const std::string& m,int t){
        return "L5|" + m + "|t" + I2S(t);
    }

    void add_crit(const std::string& k, int y){
        CritStat& st = crit_cnt_[k];
        st.n += 1; if (y) st.n1 += 1;
    }
    void add_clu(const std::string& k, int c){
        CluStat& st = clu_cnt_[k];
        st.total += 1; st.cnt[c] += 1;
    }
    void add_lat(const std::string& k, double ly){
        LatStat& st = lat_log_[k];
        st.n += 1; st.sum_log += ly;
    }

    void blend_crit(const std::string& k, double w, double& num, double& den) const {
        std::unordered_map<std::string,CritStat>::const_iterator it = crit_cnt_.find(k);
        if (it == crit_cnt_.end() || it->second.n == 0) return;
        const double a = std::max(1e-6, (double)cfg_.alpha);
        const double p = (it->second.n1 + a) / (it->second.n + 2.0 * a);
        num += w * p * (double)it->second.n;
        den += w * (double)it->second.n;
    }

    double clu_logprob(const std::string& k, int c, double a, int C) const {
        std::unordered_map<std::string,CluStat>::const_iterator it = clu_cnt_.find(k);
        if (it == clu_cnt_.end() || it->second.total == 0) return 0.0; // neutral
        std::unordered_map<int,uint64_t>::const_iterator jt = it->second.cnt.find(c);
        const double nc = (jt == it->second.cnt.end() ? 0.0 : (double)jt->second);
        const double nt = (double)it->second.total;
        return std::log((nc + a) / (nt + a * (double)C));
    }

    void blend_lat(const std::string& k, double w, double& num, double& den) const {
        std::unordered_map<std::string,LatStat>::const_iterator it = lat_log_.find(k);
        if (it == lat_log_.end() || it->second.n == 0) return;
        const double mu = it->second.sum_log / (double)it->second.n;
        num += w * mu * (double)it->second.n;
        den += w * (double)it->second.n;
    }

private:
    mutable std::mutex mu_;
    Config cfg_;

    std::unordered_map<std::string, CritStat> crit_cnt_;
    std::unordered_map<std::string, CluStat>  clu_cnt_;
    std::unordered_map<std::string, LatStat>  lat_log_;
    std::unordered_set<int> cluster_ids_;

    // unused but kept to avoid churn if referenced elsewhere
    std::unordered_map<std::string, uint64_t> crit_tot_;
    std::unordered_map<std::string, uint64_t> clu_tot_;
};
