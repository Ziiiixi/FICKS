#include "utils_sched.h"
#include "libsmctrl.h"
#include <vector>
#include <algorithm>
#include <set>  
#include "running_ops_shared.h"

cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t (*malloc_function)(void** devPtr, size_t size);
cudaError_t (*free_function)(void* devPtr);
cudaError_t (*memset_function)(void* devPtr, int  value, size_t count);
cudaError_t (*memset_async_function)(void* devPtr, int  value, size_t count, cudaStream_t stream);

cudnnStatus_t (*cudnn_create_function)(cudnnHandle_t *handle);
cudnnStatus_t (*cudnn_backend_func)(cudnnHandle_t handle, const cudnnBackendDescriptor_t executionPlan, const cudnnBackendDescriptor_t varianPack);
cudnnStatus_t (*cudnn_bnorm_reserve_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes);
cudnnStatus_t (*cudnn_conv_function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
cudnnStatus_t (*cudnn_bnorm_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
cudnnStatus_t (*cudnn_bnorm_infer_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);
cudnnStatus_t (*cudnn_rnn_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes);
cudnnStatus_t (*cudnn_rnn_train_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);

cudnnStatus_t (*cudnn_bnorm_bw_function)(
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

cudnnStatus_t (*cudnn_conv_bw_data_function)(
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

cudnnStatus_t (*cudnn_conv_bw_filter_function)(
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

cublasStatus_t (*cublas_sgemm_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
cublasStatus_t (*cublas_sgemm_strided_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);
cublasStatus_t (*cublas_lt_matmul_func)(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta, const void *C, cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream);

extern cudnnHandle_t* global_handle0;
extern cudnnHandle_t* global_handle1;

extern int status;
extern int* seen;
extern int num_tpcs;
int arrMask[24] = {};
extern uint32_t mask;
extern uint32_t *localMask;
extern uint32_t *localMask_O;
extern int* num_client_kernels;
static constexpr int HW_NUM_TPCS = 24;
extern int tpc_usage_count[24];


void process_eval(vector<vector<float>> &client_durations) {

	int num_clients = client_durations.size();
	for (int i=0; i<num_clients; i++) {
		vector<float> client_stats = client_durations[i];
		// remove first two iters
		client_stats.erase(client_stats.begin());
		client_stats.erase(client_stats.begin());

		sort(client_stats.begin(), client_stats.end());
		int client_len = client_stats.size();
		float p50 = client_stats[client_len/2];
		float p95 = client_stats[(client_len*95/100)];
		float p99 = client_stats[(client_len*99/100)];
		printf("Client %d, p50=%f, p95=%f, p99=%f\n", i, p50, p95, p99);
	}
}


void pop_from_queue(queue<struct func_record>* client_queue, pthread_mutex_t* client_mutex, int idx) {
	pthread_mutex_lock(client_mutex);
	client_queue->pop();
	pthread_mutex_unlock(client_mutex);
}

// void create_streams(cudaStream_t** sched_streams, int num, bool reef) {

//     // Allocate memory for stream priority range (although we will not use it now)
//     int* lp = (int*)malloc(sizeof(int));
//     int* hp = (int*)malloc(sizeof(int));

//     // Get the stream priority range (not needed for this modified version, but kept for clarity)
//     CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(lp, hp));

//     // Debugging: Print the highest and lowest stream priority (still retrieved but not used)
//     DEBUG_PRINT("Highest stream priority is %d, lowest stream priority is %d\n", *hp, *lp);
//     assert(*lp == 0); // Ensure the lowest priority is 0

//     // Create streams with no priority (default priority of 0) for all
//     for (int i = 0; i < num; i++) {
//         sched_streams[i] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
//         cudaStreamCreateWithPriority(sched_streams[i], cudaStreamNonBlocking, 0); // Set priority to 0
//     }
// }


void create_streams(cudaStream_t** sched_streams, int num, bool reef) {
    for (int i = 0; i < num; i++) {
        sched_streams[i] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
        CHECK_CUDA_ERROR(cudaStreamCreate(sched_streams[i]));  // plain, default flags, default priority
    }
}

// void create_streams(cudaStream_t** sched_streams, int num, bool reef) {

// 	int* lp = (int*)malloc(sizeof(int));
// 	int* hp = (int*)malloc(sizeof(int));

// 	CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(lp, hp));

// 	DEBUG_PRINT("Highest stream priority is %d, lowest stream priority is %d\n", *hp, *lp);
// 	assert(*lp==0);

// 	for (int i=0; i<num-1; i++) {
// 		sched_streams[i] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
// 		cudaStreamCreateWithPriority(sched_streams[i], cudaStreamNonBlocking, 0);
// 	}

// 	// client num-1 is high priority
// 	if (!reef) {
// 		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
// 		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, *hp);
// 	}
// 	else {
// 		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
// 		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, 0);
// 	}

// }

// void create_streams(cudaStream_t** sched_streams, int num, bool reef) {

// 	int* lp = (int*)malloc(sizeof(int));
// 	int* hp = (int*)malloc(sizeof(int));

// 	CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(lp, hp));

// 	DEBUG_PRINT("Highest stream priority is %d, lowest stream priority is %d\n", *hp, *lp);
// 	assert(*lp==0);

// 	for (int i=0; i<num-1; i++) {
// 		sched_streams[i] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
// 		cudaStreamCreateWithPriority(sched_streams[i], cudaStreamNonBlocking, 0);
// 	}

// 	// client num-1 is high priority
// 	if (!reef) {
// 		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
// 		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, *hp);
// 	}
// 	else {
// 		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
// 		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, 0);
// 	}

// }

void create_events(cudaEvent_t*** events, int num) {

	// per-stream event
	for (int i=0; i<num; i++) {
		events[i] = (cudaEvent_t**)malloc(30000*sizeof(cudaEvent_t*));
		for (int j=0; j<30000; j++) {
			//printf("create %d, %d\n", i, j);
			events[i][j] = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
			CHECK_CUDA_ERROR(cudaEventCreateWithFlags(events[i][j], cudaEventDisableTiming));
		}
	}
}

void register_functions() {

    // for kernel
	*(void **)(&kernel_function) = dlsym(RTLD_DEFAULT, "cudaLaunchKernel");
	assert(kernel_function != NULL);

	// for memcpy
	*(void **)(&memcpy_function) = dlsym (RTLD_DEFAULT, "cudaMemcpy");
	assert(memcpy_function != NULL);

	// for memcpy_async
	*(void **)(&memcpy_async_function) = dlsym (RTLD_DEFAULT, "cudaMemcpyAsync");
	assert(memcpy_async_function != NULL);

	// for malloc
	*(void **)(&malloc_function) = dlsym (RTLD_DEFAULT, "cudaMalloc");
	assert(malloc_function != NULL);

	// for free
	*(void **)(&free_function) = dlsym (RTLD_DEFAULT, "cudaFree");
	assert(free_function != NULL);

	// for memset
	*(void **)(&memset_function) = dlsym (RTLD_DEFAULT, "cudaMemset");
	assert (memset_function != NULL);

	// for memset_async
	*(void **)(&memset_async_function) = dlsym (RTLD_DEFAULT, "cudaMemsetAsync");
	assert (memset_async_function != NULL);

	// for cudnn create
	*(void **)(&cudnn_create_function) = dlsym(RTLD_DEFAULT, "cudnnCreate");
	assert(cudnn_create_function != NULL);

	// for cudnn_bnorm_reserve
	*(void **)(&cudnn_bnorm_reserve_function) = dlsym(RTLD_DEFAULT, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
	assert(cudnn_bnorm_reserve_function != NULL);

	// for cudnn conv
	*(void **)(&cudnn_backend_func) = dlsym(RTLD_DEFAULT, "cudnnBackendExecute");
	assert(cudnn_backend_func != NULL);

	// for cudnn conv
	*(void **)(&cudnn_conv_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionForward");
	assert(cudnn_conv_function != NULL);

	// for bnorm train
	*(void **)(&cudnn_bnorm_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardTrainingEx");
	assert(cudnn_bnorm_function != NULL);

	// for bnorm infer
	*(void **)(&cudnn_bnorm_infer_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardInference");
	assert(cudnn_bnorm_infer_function != NULL);

	// for bnorm backward
	*(void **)(&cudnn_bnorm_bw_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationBackwardEx");
	assert(cudnn_bnorm_bw_function != NULL);

	// for conv data backward
	*(void **)(&cudnn_conv_bw_data_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionBackwardData");
	assert(cudnn_conv_bw_data_function != NULL);

	// for conv filter backward
	*(void **)(&cudnn_conv_bw_filter_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionBackwardFilter");
	assert(cudnn_conv_bw_filter_function != NULL);

	// CUBLAS sgemm
	*(void **)(&cublas_sgemm_function) = dlsym(RTLD_DEFAULT, "cublasSgemm_v2");
	assert(cublas_sgemm_function != NULL);

	// CUBLAS sgemm strided
	*(void **)(&cublas_sgemm_strided_function) = dlsym(RTLD_DEFAULT, "cublasSgemmStridedBatched");
	assert(&cublas_sgemm_strided_function != NULL);

	// CUBLAS matmul
	*(void **)(&cublas_lt_matmul_func) = dlsym(RTLD_DEFAULT, "cublasLtMatmul");
	assert(&cublas_lt_matmul_func != NULL);
}

void schedule_kernel(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid) {

	switch (frecord.type) {
		case KERNEL_RECORD: {
			//DEBUG_PRINT("found a new kernel record from idx %d! kernel func is %p\n", idx, kernel_function);
			kernel_record record = frecord.data.krecord;
			cudaError status = (*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, *sched_stream);
			assert(status == cudaSuccess);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case MEMCPY_RECORD: {
			memcpy_record record = frecord.data.mrecord;
			if (!record.async) {
				(*memcpy_function)(record.dst, record.src, record.count, record.kind);
			} else {
				(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, *sched_stream);
			}
			break;
		}
		case MALLOC_RECORD: {
			DEBUG_PRINT("found a new malloc record from idx %d!\n", idx);
			malloc_record record = frecord.data.malrecord;
			(*malloc_function)(record.devPtr, record.size);
			break;
		}
		case FREE_RECORD: {
			DEBUG_PRINT("found a new FREE record from idx %d!\n", idx);
			free_record record = frecord.data.frecord;
			//(*free_function)(record.devPtr);
			break;
		}
		case MEMSET_RECORD: {
			memset_record record = frecord.data.msetrecord;
			if (record.async) {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_async_function)(record.devPtr, record.value, record.count, *sched_stream);
			}
			else {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_function)(record.devPtr, record.value, record.count);
			}
			break;
		}
		case CUDNN_BACKEND_RECORD: {
			cudnnBackend_record record = frecord.data.cudnnBackendRecord;
			cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
			//printf("found a new cudnn backend record, handle is %p, stream is %d!\n", record.handle, *sched_stream);
			status = cudnnSetStream(record.handle, *sched_stream);
			if (status != CUDNN_STATUS_SUCCESS) {
				fprintf(stderr, "cudnnSetStream failed: %s (%d), handle=%p stream=%p\n",
						cudnnGetErrorString(status), (int)status,
						(void*)record.handle, (void*)*sched_stream);
				fflush(stderr);
				abort();
			}
			assert (status == CUDNN_STATUS_SUCCESS);
			status = (*cudnn_backend_func)(record.handle, record.executionPlan, record.varianPack);
			assert (status == CUDNN_STATUS_SUCCESS);

			//CHECK_CUDA_ERROR(cudaStreamSynchronize(*sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_CONV_RECORD: {

			DEBUG_PRINT("found a new cudnn conv record from idx %d!\n", idx);
			cudnnConvolutionForward_record record = frecord.data.cudnnConvRecord;
			cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
			status = cudnnSetStream(record.handle, *sched_stream);
			assert (status == CUDNN_STATUS_SUCCESS);
			status = (*cudnn_conv_function)(record.handle, record.alpha, record.xDesc, record.x, record.wDesc, record.w, record.convDesc, record.algo, record.workSpace, record.workSpaceSizeInBytes, record.beta, record.yDesc, record.y);
			assert (status == CUDNN_STATUS_SUCCESS);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_BNORM_RECORD: {
			//DEBUG_PRINT("found a new bnorm training record from idx %d!\n", idx);
			cudnnBatchNormalizationForwardTrainingEx_record record = frecord.data.cudnnBNormRecord;

			//printf("Got a CUDNN operation from client %d, handle is %p\n", idx, record.handle);
			cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

			//printf("Stream is %d\n", *sched_stream);
			status = cudnnSetStream(record.handle, *sched_stream);
			assert (status == CUDNN_STATUS_SUCCESS);
			status = (*cudnn_bnorm_function)(
				record.handle,
				record.mode,
				record.bnOps,
				record.alpha,
				record.beta,
				record.xDesc,
				record.xData,
				record.zDesc,
				record.zData,
				record.yDesc,
				record.yData,
				record.bnScaleBiasMeanVarDesc,
				record.bnScaleData,
				record.bnBiasData,
				record.exponentialAverageFactor,
				record.resultRunningMeanData,
				record.resultRunningVarianceData,
				record.epsilon,
				record.saveMean,
				record.saveInvVariance,
				record.activationDesc,
				record.workspace,
				record.workSpaceSizeInBytes,
				record.reserveSpace,
				record.reserveSpaceSizeInBytes
			);
			assert (status == CUDNN_STATUS_SUCCESS);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_BNORM_INF_RECORD: {
			DEBUG_PRINT("found a new bnorm inf record from idx %d!\n", idx);
			cudnnBatchNormalizationForwardInference_record record = frecord.data.cudnnBNormInfRecord;
			//printf("Got a CUDNN operation from client %d, handle is %p\n", idx, record.handle);
			cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
			status = cudnnSetStream(record.handle, *sched_stream);
			assert (status == CUDNN_STATUS_SUCCESS);

			// cudaStream_t cur_stream;
			// status = cudnnGetStream(record.handle, &cur_stream);
			// assert (status == CUDNN_STATUS_SUCCESS);
			// if (cur_stream != *sched_stream) {
			// 	status = cudnnSetStream(record.handle, *sched_stream);
			// 	assert (status == CUDNN_STATUS_SUCCESS);
			// }
			status = (*cudnn_bnorm_infer_function)(record.handle, record.mode, record.alpha, record.beta, record.xDesc, record.x, record.yDesc, record.y, record.bnScaleBiasMeanVarDesc, record.bnScale, record.bnBias, record.estimatedMean, record.estimatedVariance, record.epsilon);
			assert (status == CUDNN_STATUS_SUCCESS);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_RNN_INF_RECORD: {
			DEBUG_PRINT("found a new cudnn rnn inf record from idx %d!\n", idx);
			cudnnRNNForwardInf_record record = frecord.data.cudnnRnnInfRecord;
			cudnnSetStream(record.handle, *sched_stream);
			cudnnStatus_t status = (*cudnn_rnn_function)(record.handle, record.rnnDesc, record.seqLength, record.xDesc, record.x, record.hxDesc, record.hx, record.cxDesc, record.cx, record.wDesc, record.w, record.yDesc, record.y, record.hyDesc, record.hy, record.cyDesc, record.cy, record.workspace, record.workSpaceSizeInBytes);
			assert (status == CUDNN_STATUS_SUCCESS);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_RNN_TRAIN_RECORD: {
			DEBUG_PRINT("found a new cudnn rnn train record from idx %d!\n", idx);
			cudnnRNNForwardTraining_record record = frecord.data.cudnnRnnTrainRecord;
			cudnnSetStream(record.handle, *sched_stream);
			cudnnStatus_t status = (*cudnn_rnn_train_function)(
				record.handle,
				record.rnnDesc,
				record.seqLength,
				record.xDesc,
				record.x,
				record.hxDesc,
				record.hx,
				record.cxDesc,
				record.cx,
				record.wDesc,
				record.w,
				record.yDesc,
				record.y,
				record.hyDesc,
				record.hy,
				record.cyDesc,
				record.cy,
				record.workspace,
				record.workSpaceSizeInBytes,
				record.reserveSpace,
				record.reserveSpaceSizeInBytes
			);
			assert (status == CUDNN_STATUS_SUCCESS);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_BNORM_BACKWARD_RECORD: {
			DEBUG_PRINT("found a new cudnn batch norm backw record from idx %d!\n", idx);

			cudnnBatchNormalizationBackwardEx_record record = frecord.data.cudnnBNormBackRecord;
			cudnnSetStream(record.handle, *sched_stream);
			cudnnStatus_t status = (*cudnn_bnorm_bw_function)(
					record.handle,
					record.mode,
					record.bnOps,
					record.alphaDataDiff,
					record.betaDataDiff,
					record.alphaParamDiff,
					record.betaParamDiff,
					record.xDesc,
					record.xData,
					record.yDesc,
					record.yData,
					record.dyDesc,
					record.dyData,
					record.dzDesc,
					record.dzData,
					record.dxDesc,
					record.dxData,
					record.dBnScaleBiasDesc,
					record.bnScaleData,
					record.bnBiasData,
					record.dBnScaleData,
					record.dBnBiasData,
					record.epsilon,
					record.savedMean,
					record.savedInvVariance,
					record.activationDesc,
					record.workspace,
					record.workSpaceSizeInBytes,
					record.reserveSpace,
					record.reserveSpaceSizeInBytes
			);
			assert (status == CUDNN_STATUS_SUCCESS);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_CONV_DATA_RECORD: {
			DEBUG_PRINT("found a new cudnn conv data backw record from idx %d!\n", idx);

			cudnnConvolutionBackwardData_record record = frecord.data.cudnnConvBackDataRecord;
			cudnnSetStream(record.handle, *sched_stream);
			DEBUG_PRINT("submit!\n");
			cudnnStatus_t status = (*cudnn_conv_bw_data_function)(
					record.handle,
					record.alpha,
					record.wDesc,
					record.w,
					record.dyDesc,
					record.dy,
					record.convDesc,
					record.algo,
					record.workSpace,
					record.workSpaceSizeInBytes,
					record.beta,
					record.dxDesc,
					record.dx
			);
			assert (status == CUDNN_STATUS_SUCCESS);

			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_CONV_FILTER_RECORD: {
			DEBUG_PRINT("found a new cudnn conv filter backw record from idx %d!\n", idx);

			cudnnConvolutionBackwardFilter_record record = frecord.data.cudnnConvBackFilterRecord;
			cudnnSetStream(record.handle, *sched_stream);
			cudnnStatus_t status = (*cudnn_conv_bw_filter_function)(
					record.handle,
					record.alpha,
					record.xDesc,
					record.x,
					record.dyDesc,
					record.dy,
					record.convDesc,
					record.algo,
					record.workSpace,
					record.workSpaceSizeInBytes,
					record.beta,
					record.dwDesc,
					record.dw
			);
			assert (status == CUDNN_STATUS_SUCCESS);

			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm record from idx %d!\n", idx);

			cublasSgemm_record record = frecord.data.cublasSgemmRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			cublasStatus_t status = (*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
			//cublasSetStream_v2(record.handle, 0);
			assert (status == CUBLAS_STATUS_SUCCESS);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_STRIDED_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm strided record from idx %d!\n", idx);

			cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			cublasStatus_t status = (*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
			//cublasSetStream_v2(record.handle, 0);
			assert (status == CUBLAS_STATUS_SUCCESS);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_MATMUL_RECORD: {
			DEBUG_PRINT("found a new cublas matmul record from idx %d, func is %p!\n", idx, cublas_lt_matmul_func);
			cublasLtMatmul_record record = frecord.data.cublasLtMatmulRecord;
			cublasStatus_t status = (*cublas_lt_matmul_func)(
				record.lightHandle,
				record.computeDesc,
				record.alpha,
				record.A,
				record.Adesc,
				record.B,
				record.Bdesc,
				record.beta,
				record.C,
				record.Cdesc,
				record.D,
				record.Ddesc,
				record.algo,
				record.workspace,
				record.workspaceSizeInBytes,
				*sched_stream
			);
			assert (status == CUBLAS_STATUS_SUCCESS);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		default:
			printf("UNSUPPORTED OPERATION - ABORT\n");
			abort();

	}
	//DEBUG_PRINT("Return from schedule, seen[%d] is %d!\n", idx, seen[idx]);
	CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
	//event_ids[evid] += 1;
}

void schedule_kernel_profile(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid, op_info op_info_cur, int tpc_usage, int cur_iter) {
	
	switch (frecord.type) {
		case KERNEL_RECORD: {
			DEBUG_PRINT("found a new kernel record from idx %d! kernel func is %p\n", idx, kernel_function);
			kernel_record record = frecord.data.krecord;
			
			// Initialize CUDA events
			cudaEvent_t start, stop;
			cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
			cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);
			
			// Capture the host start time in nanoseconds
			// auto host_start = std::chrono::high_resolution_clock::now();
			// auto host_start_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(host_start).time_since_epoch().count();
			
			// Record CUDA event start
			cudaEventRecord(start, *sched_stream);
			
			// Launch the kernel
			(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, *sched_stream);
			
			// Record CUDA event stop
			cudaEventRecord(stop, *sched_stream);
			
			// Synchronize to ensure kernel completion
			cudaEventSynchronize(stop);

			// unsetmask_nomutex(idx);

			// Capture the host finish time in nanoseconds
			// auto host_finish = std::chrono::high_resolution_clock::now();
			// auto host_finish_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(host_finish).time_since_epoch().count();
			
			// Calculate elapsed time using CUDA events
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			long long nanoseconds = static_cast<long long>(milliseconds * 1e6);
			
			// Output the information with nanosecond timestamps
			// if(cur_iter > 9){
			// 	std::cout << "Name of kernel: " << op_info_cur.name
			// 			<< " | Current iter: " << cur_iter
			// 			<< " | Client ID: " << idx
			// 			<< " | Grid Size: " << op_info_cur.grid
			// 			<< " | Block Size: " << op_info_cur.block
			// 			<< " | TPC Usage: " << tpc_usage
			// 			<< " | Critical: " << 1
			// 			<< " | Kernel Index: " << seen[idx]
			// 			// << " | Kernel Index: " << event_ids[evid]
			// 			<< " | Knee TPC: " << op_info_cur.knee_tpc
			// 			<< " | Kernel execution time: " << nanoseconds << " ns"
			// 			<< std::endl;
			// }
			
			// Clean up CUDA events
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			
			// Update event and seen counts
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case MEMCPY_RECORD: {
			memcpy_record record = frecord.data.mrecord;
			if (!record.async) {
				(*memcpy_function)(record.dst, record.src, record.count, record.kind);
			} else {
				(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, *sched_stream);
			}
			break;
		}
		case MALLOC_RECORD: {
			DEBUG_PRINT("found a new malloc record from idx %d!\n", idx);
			malloc_record record = frecord.data.malrecord;
			(*malloc_function)(record.devPtr, record.size);
			break;
		}
		case FREE_RECORD: {
			DEBUG_PRINT("found a new FREE record from idx %d!\n", idx);
			free_record record = frecord.data.frecord;
			//(*free_function)(record.devPtr);
			break;
		}
		case MEMSET_RECORD: {
			memset_record record = frecord.data.msetrecord;
			if (record.async) {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_async_function)(record.devPtr, record.value, record.count, *sched_stream);
			}
			else {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_function)(record.devPtr, record.value, record.count);
			}
			break;
		}
		case CUBLAS_SGEMM_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm record from idx %d!\n", idx);
			cublasSgemm_record record = frecord.data.cublasSgemmRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			cudaEvent_t start, stop;
			cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
			cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);

			cudaEventRecord(start, *sched_stream);
			(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
			cudaEventRecord(stop, *sched_stream);
			cudaEventSynchronize(stop); 

			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			long long nanoseconds = static_cast<long long>(milliseconds * 1e6);
			
			// if(cur_iter > 9){
			// 	std::cout << "Name of kernel: " << op_info_cur.name
			// 			<< " | Current iter: " << cur_iter
			// 			<< " | Client ID: " << idx
			// 			<< " | Grid Size: " << op_info_cur.grid
			// 			<< " | Block Size: " << op_info_cur.block
			// 			<< " | TPC Usage: " << tpc_usage
			// 			<< " | Critical: " << 1
			// 			<< " | Kernel Index: " << seen[idx]
			// 			// << " | Kernel Index: " << event_ids[evid]
			// 			<< " | Knee TPC: " << op_info_cur.knee_tpc
			// 			<< " | Kernel execution time: " << nanoseconds << " ns"
			// 			// << " | Start Time: " << host_start_ns << " ns since epoch"
			// 			// << " | Finish Time: " << host_finish_ns << " ns since epoch"
			// 			<< std::endl;
			// }

			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_STRIDED_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm strided record from idx %d!\n", idx);
			cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
			cudaEvent_t start, stop;
			cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
			cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);

			cudaEventRecord(start, *sched_stream);
			(*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
			cublasSetStream_v2(record.handle, *sched_stream);
			cudaEventRecord(stop, *sched_stream);
			cudaEventSynchronize(stop); 

			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			long long nanoseconds = static_cast<long long>(milliseconds * 1e6);
			
			// if(cur_iter > 9){
			// 	std::cout << "Name of kernel: " << op_info_cur.name
			// 			<< " | Current iter: " << cur_iter
			// 			<< " | Client ID: " << idx
			// 			<< " | Grid Size: " << op_info_cur.grid
			// 			<< " | Block Size: " << op_info_cur.block
			// 			<< " | TPC Usage: " << tpc_usage
			// 			<< " | Critical: " << 1
			// 			<< " | Kernel Index: " << seen[idx]
			// 			// << " | Kernel Index: " << event_ids[evid]
			// 			<< " | Knee TPC: " << op_info_cur.knee_tpc
			// 			<< " | Kernel execution time: " << nanoseconds << " ns"
			// 			// << " | Start Time: " << host_start_ns << " ns since epoch"
			// 			// << " | Finish Time: " << host_finish_ns << " ns since epoch"
			// 			<< std::endl;
			// }

			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		default:
			printf("UNSUPPORTED OPERATION - ABORT\n");
			abort();
			// abort(1, 0, "UNSUPPORTED OPERATION - ABORT\n");

	}
	DEBUG_PRINT("Return from schedule, seen[%d] is %d!\n", idx, seen[idx]);
	CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
	//event_ids[evid] += 1;
}

void setmask_krisp(int num_tpcs_need, int idx) {
    // pthread_mutex_lock(client_mutex);

    // Print the mask before setting
    // std::cout << "Mask before setting: " << std::bitset<32>(mask) << std::endl;
    // std::cout << "localMask[" << idx << "] before setting: " << std::bitset<32>(localMask[idx]) << std::endl;

    int count = 0;
    for (int i = 0; i < 24 && count < num_tpcs_need; ++i) {
        if ((mask & (1 << i)) == 0) { // Check if the bit is 0
            mask |= (1 << i); // Set the bit to 1
            localMask[idx] |= (1 << i);
			localMask_O[idx] |= (1 << i);
			tpc_usage_count[i]++;
            ++count;
        }
    }


    if (count < num_tpcs_need) {
        std::cerr << "Error: Insufficient 0 bits available in the mask to set " 
                  << num_tpcs_need << " bits." << std::endl;
        abort();
    }


    num_tpcs -= count;
    uint32_t mask_to_set = ~localMask[idx];
	
	// std::cout << "[setmask] client " << idx << ", need " << num_tpcs_need << "\n";

	// std::cout << "  mask_to_set   (bits 31..0): ";
	// for (int b = 31; b >= 0; --b) {
	// 	std::cout << (int)((mask_to_set >> b) & 1u);
	// 	if (b % 8 == 0) std::cout << ' ';
	// }
	// std::cout << "\n";
    libsmctrl_set_KRISP_I_mask(mask_to_set);
}



void setmask(int num_tpcs_need, int idx){
    int count = 0;
	localMask_O[idx] = 0;

	int tpc_indices[24];
    for (int i = 0; i < 24; ++i) {
        tpc_indices[i] = i;
    }

	std::sort(tpc_indices, tpc_indices + 24, [](int a, int b) {
        return tpc_usage_count[a] < tpc_usage_count[b];
    });

    for (int i = 0; i < 24 && count < num_tpcs_need; ++i) {
        int tpc = tpc_indices[i];
		localMask_O[idx] |= (1 << tpc); 
		tpc_usage_count[tpc]++;
		++count;
    }
	// std::cout << "Mask after setting: " << std::bitset<32>(mask) << std::endl;
    // std::cout << "localMask[" << idx << "] after setting: " << std::bitset<32>(localMask_O[idx]) << std::endl;
	uint32_t mask_to_set = ~localMask_O[idx];
	libsmctrl_set_KRISP_I_mask(mask_to_set);
	// libsmctrl_set_mask();
}


void setmask_O(int num_tpcs_need, int idx){
    int count = 0;
	localMask_O[idx] = 0;

	int tpc_indices[24];
    for (int i = 0; i < 24; ++i) {
        tpc_indices[i] = i;
    }

	std::sort(tpc_indices, tpc_indices + 24, [](int a, int b) {
        return tpc_usage_count[a] < tpc_usage_count[b];
    });

    for (int i = 0; i < 24 && count < num_tpcs_need; ++i) {
        int tpc = tpc_indices[i];
		localMask_O[idx] |= (1 << tpc); 
		tpc_usage_count[tpc]++;
		++count;
    }
	// std::cout << "Mask after setting: " << std::bitset<32>(mask) << std::endl;
    // std::cout << "localMask[" << idx << "] after setting: " << std::bitset<32>(localMask_O[idx]) << std::endl;
	uint32_t mask_to_set = ~localMask_O[idx];
	// libsmctrl_set_KRISP_I_mask(mask_to_set);
}


// Helpers
// ------------------------------------------------------------
// static inline uint32_t all_tpcs_mask_u32() {
//     return (HW_NUM_TPCS == 32) ? 0xFFFFFFFFu : ((1u << HW_NUM_TPCS) - 1u);
// }

// static inline int popcnt_u32(uint32_t x) {
//     const uint32_t ALL = all_tpcs_mask_u32();
//     return __builtin_popcount(x & ALL);
// }

// // Pick k bits from pool, preferring lower tpc_usage_count, tie by smaller tpc id.
// static inline uint32_t pick_k_bits_from_pool(uint32_t pool, int k) {
//     const uint32_t ALL = all_tpcs_mask_u32();
//     pool &= ALL;
//     if (k <= 0 || pool == 0) return 0u;

//     struct Cand { int tpc; int score; };
//     std::vector<Cand> cands;
//     cands.reserve(HW_NUM_TPCS);

//     for (int t = 0; t < HW_NUM_TPCS; ++t) {
//         if (pool & (1u << t)) {
//             cands.push_back({t, tpc_usage_count[t]});
//         }
//     }

//     std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
//         if (a.score != b.score) return a.score < b.score;
//         return a.tpc < b.tpc;
//     });

//     uint32_t out = 0u;
//     int take = std::min(k, (int)cands.size());
//     for (int i = 0; i < take; ++i) {
//         out |= (1u << cands[i].tpc);
//     }
//     return out;
// }

// static inline void apply_subset_mask_u32(uint32_t subset) {
//     const uint32_t ALL = all_tpcs_mask_u32();
//     subset &= ALL;
//     const uint32_t mask_to_set = (~subset) & ALL;
//     libsmctrl_set_KRISP_I_mask(mask_to_set);
// }

// // Release all owned bits for one client and decrement usage counters.
// static inline void release_client_ownership_O(int idx) {
//     uint32_t owned = localMask_O[idx] & all_tpcs_mask_u32();
//     if (!owned) return;

//     for (int t = 0; t < HW_NUM_TPCS; ++t) {
//         if (owned & (1u << t)) {
//             if (tpc_usage_count[t] > 0) --tpc_usage_count[t];
//             ++num_tpcs; // keep only if num_tpcs means "free tpcs" in your code
//         }
//     }
//     localMask_O[idx] = 0u;
// }

// // Optional: call this when you truly want to clear this client's ownership and unmask all.
// static inline void unsetmask_O_and_release(int idx) {
//     release_client_ownership_O(idx);
//     apply_subset_mask_u32(all_tpcs_mask_u32()); // all active
// }

// ------------------------------------------------------------
// Main allocator
// return: subset bits actually granted for THIS launch
//         0 means "failed" (strict long admission failure)
// // ------------------------------------------------------------
// uint32_t setmask_O(int total_need, int idx, bool is_long)
// {
//     constexpr int TPC_TOTAL = HW_NUM_TPCS;
//     const uint32_t ALL = all_tpcs_mask_u32();

//     // no-mask mode => use all tpcs
//     if (total_need <= 0) total_need = TPC_TOTAL;
//     if (total_need > TPC_TOTAL) total_need = TPC_TOTAL;

//     // ------------------------------------------------------------
//     // 1) Build forbidden mask for LONG kernels:
//     //    forbidden = union of other clients' currently running LONG subsets
//     // ------------------------------------------------------------
//     uint32_t forbidden = 0u;
//     if (is_long) {
//         for (const auto& e : g_running_ops) {
//             const op_info* op = e.op;
//             if (!op) continue;
//             if (!op->is_running) continue;
//             if (e.client_id == idx) continue;
//             if (op->is_short == 1) continue; // only block against other LONG kernels

//             uint32_t sub = op->subset_mask & ALL;

//             // robust fallback for no-mask/full-device launches
//             if (sub == 0u && op->tpc_used >= TPC_TOTAL) {
//                 sub = ALL;
//             }

//             forbidden |= sub;
//         }
//     }

//     // long: strict non-overlap against forbidden
//     // short: allow overlap, so full pool
//     const uint32_t allowed = is_long ? (ALL & ~forbidden) : ALL;

//     if (is_long && popcnt_u32(allowed) < total_need) {
//         // strict long admission failure
//         return 0u;
//     }

//     // ------------------------------------------------------------
//     // 2) Current ownership and usable ownership
//     // ------------------------------------------------------------
//     uint32_t owned = localMask_O[idx] & ALL;
//     uint32_t owned_usable = owned & allowed;

//     // First try from already-owned usable bits
//     uint32_t subset = pick_k_bits_from_pool(owned_usable, total_need);
//     int have = popcnt_u32(subset);

//     // ------------------------------------------------------------
//     // 3) Need to expand ownership
//     // ------------------------------------------------------------
//     if (have < total_need) {
//         int need_add = total_need - have;

//         // prefer new bits from allowed and not already owned
//         uint32_t add_pool = allowed & ~owned;
//         int add_pool_cnt = popcnt_u32(add_pool);

//         if (is_long && add_pool_cnt < need_add) {
//             // strict long: cannot satisfy
//             return 0u;
//         }

//         if (!is_long && add_pool_cnt < need_add) {
//             need_add = add_pool_cnt; // short can degrade
//         }

//         uint32_t add_bits = (need_add > 0) ? pick_k_bits_from_pool(add_pool, need_add) : 0u;
//         add_bits &= ~owned; // only truly new ownership

//         // update ownership + usage counters
//         if (add_bits) {
//             for (int t = 0; t < TPC_TOTAL; ++t) {
//                 if (add_bits & (1u << t)) {
//                     localMask_O[idx] |= (1u << t);
//                     ++tpc_usage_count[t];
//                     --num_tpcs; // keep only if num_tpcs means "free tpcs"
//                 }
//             }
//         }

//         // rebuild subset from (owned & allowed)
//         owned = localMask_O[idx] & ALL;
//         owned_usable = owned & allowed;
//         subset = pick_k_bits_from_pool(owned_usable, total_need);
//     }

//     // strict long must be exact
//     if (is_long && popcnt_u32(subset) != total_need) {
//         return 0u;
//     }

//     // short fallback: if still empty (rare), borrow from allowed directly
//     if (!is_long && popcnt_u32(subset) == 0 && total_need > 0) {
//         subset = pick_k_bits_from_pool(allowed, total_need);
//     }

//     // Apply runtime mask
//     apply_subset_mask_u32(subset);
//     return subset;
// }


// void unsetmask(int idx) {
//     constexpr uint32_t ALL = (24 == 32) ? 0xFFFFFFFFu : ((1u << 24) - 1u);

//     uint32_t owned = localMask_O[idx] & ALL;

//     for (int i = 0; i < 24; ++i) {
//         if (owned & (1u << i)) {
//             if (tpc_usage_count[i] <= 0) {
//                 printf("[ERR][unsetmask] tpc_usage_count[%d] already <= 0 (idx=%d)\n", i, idx);
//                 abort();
//             }
//             --tpc_usage_count[i];
//         }
//     }

//     // clear ownership for this client
//     localMask_O[idx] = 0u;

//     // optional: if your runtime expects explicit "fully unmask now"
//     // libsmctrl_set_KRISP_I_mask(0u);
// }


void unsetmask(int idx){
	int count = 0;
	for (int i = 0; i < 24; ++i) {
        if ((localMask[idx] & (1 << i)) != 0) {
            if ((mask & (1 << i)) != 0) {
                mask &= ~(1 << i);
				count++;
            }
			else{
				printf("error setting the mask\n");
				abort();
			}
        }
    }

	// for (int i = 0; i < 24; ++i) {
    //     if (localMask_O[idx] & (1 << i)) {
    //         tpc_usage_count[i]--; 
    //         localMask_O[idx] &= ~(1 << i);
    //     }
    // }
	
	// std::cout << "mask after unset" << std::bitset<32>(mask) <<" from idx: "<<idx<< std::endl;
	localMask[idx] = 0;
	num_tpcs += count;
}