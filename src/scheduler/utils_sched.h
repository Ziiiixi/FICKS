#include <cuda_runtime.h>
#include <cuda.h>
// You have this in scheduler codex
#include "../system_utils.h"
#include "../cuda_capture/intercept_temp.h"

extern cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
extern cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t (*malloc_function)(void** devPtr, size_t size);
extern cudaError_t (*free_function)(void* devPtr);
extern cudaError_t (*memset_function)(void* devPtr, int  value, size_t count);
extern cudaError_t (*memset_async_function)(void* devPtr, int  value, size_t count, cudaStream_t stream);

extern cudnnStatus_t (*cudnn_create_function)(cudnnHandle_t *handle);
extern cudnnStatus_t (*cudnn_bnorm_reserve_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes);
extern cudnnStatus_t (*cudnn_conv_function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
extern cudnnStatus_t (*cudnn_bnorm_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
extern cudnnStatus_t (*cudnn_bnorm_infer_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);
extern cudnnStatus_t (*cudnn_rnn_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes);
extern cudnnStatus_t (*cudnn_rnn_train_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
extern cublasStatus_t (*cublas_sgemm_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
extern cublasStatus_t (*cublas_sgemm_strided_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);
extern cublasStatus_t (*cublas_lt_matmul_func)(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta, const void *C, cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream);

extern cudnnStatus_t (*cudnn_bnorm_bw_function)(
	cudnnHandle_t handle,
	cudnnBatchNormMode_t mode,
	cudnnBatchNormOps_t bnOps,
	const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData,
    const void *bnBiasData,
    void *dBnScaleData,
    void *dBnBiasData,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance,
    const cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
);

extern cudnnStatus_t (*cudnn_conv_bw_data_function)(
	cudnnHandle_t handle,
    const void *alpha,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
);

extern cudnnStatus_t (*cudnn_conv_bw_filter_function)(
	cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw
);

struct op_info {
    // -------- legacy fields --------
    std::string name;
    int profile   = -1;
    int mem       = 0;
    int sm_used   = 0;
    float duration = 0.0f;     // ms
    int grid      = 0;
    int block     = 0;
    int knee_tpc  = 0;
    int is_on_list = 0;
    float ratio_ = 0.0f;
    int tpc_used  = 0;

    // -------- FICKS fields --------
    int id                    = -1;
    int cluster               = -1;
    int opt_tpc_colocated     = -1;
    int opt_tpc_exclusive     = -1;
    int is_short              = -1; // GT: 1 short, 0 long
    std::vector<float> excl_us_by_n; // GT latency by TPC (index=tpc)
    bool     is_running       = false;
    uint64_t est_start_ns     = 0;
    uint64_t est_finish_ns    = 0;
    uint32_t subset_mask      = 0;

    // -------- predicted outputs --------
    float p_critical          = 0.5f;
    int   pred_is_critical    = 0;   // 1 critical(long), 0 non critical(short)

    int   pred_cluster        = -1;
    float p_cluster           = 0.0f;

    std::vector<float> pred_excl_us_by_n; // predicted latency by TPC
    float pred_excl_full_us   = NAN;      // predicted full GPU latency (e.g. TPC=24)
    float pred_excl_mape      = NAN;      // optional eval cache

    op_info() = default;
};


// same header
enum AlgoId {
    ALGO_ORION       = 0,
    ALGO_REEF        = 1,
    ALGO_MULTI_STREAM= 2,
    ALGO_KRISP       = 3,
    ALGO_FICKS       = 4,  // reserve
};


void register_functions();
void schedule_kernel(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid);
void schedule_kernel_profile(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid, op_info op_info_cur, int tpc_usage, int cur_iter);
void schedule_kernel_KRISP(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid, vector<int> num_clients);
void schedule_kernel_KRISP_I(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid, pthread_mutex_t* client_mutex);

void schedule_pair(
	vector<func_record*> &frecords,
	queue<struct func_record>** &buffers,
	pthread_mutex_t** &mutexes,
	vector<vector<op_info>> &op_info_vector,
	int* seen, int max_sms,
	cudaStream_t** sched_streams,
	int* streams,
	cudaEvent_t*** events,
	int num_events,
	int* event_ids
);
void schedule_pair_kernel_padding(
	vector<func_record*> &frecords,
	queue<struct func_record>** &cbuffers,
	pthread_mutex_t** &cmutexes,
	vector<vector<op_info>> &op_info_vector,
	int* seen, int max_sms,
	cudaStream_t** sched_streams,
	int* streams,
	cudaEvent_t*** events,
	int num_events,
	int* event_ids
);
void pop_from_queue(queue<struct func_record>* client_queue, pthread_mutex_t* client_mutex, int idx);
void setmask(int num_tpcs_need, int idx);
void setmask_O(int num_tpcs_need, int idx);
void setmask_krisp(int num_tpcs_need, int idx);
// uint32_t setmask_O(int total_need, int idx, bool is_long);
void unsetmask(int idx);
void create_streams(cudaStream_t** sched_streams, int num, bool reef);
void create_events(cudaEvent_t*** events, int num);
void wait_for_stream(int idx, int profile, int current_prio, int prev_prio, cudaStream_t* sched_stream, cudaEvent_t*** events, int num_events, int* event_ids);
void wait_all_streams(int idx, cudaStream_t* sched_stream, cudaEvent_t*** events, int num_events, int* event_ids);
void process_eval(vector<vector<float>> &client_durations);