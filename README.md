# Orion

This implementation is based on project orion

## Project Structure
```
> tree .
├── profiling                     # Scripts and instructions for profiling
│   ├── benchmarks                # Scripts of DNN models for profiling
│   ├── postprocessing            # Scripts for processing of profile files
└── src                           # Source code
│   ├── cuda_capture              # Code to intercept CUDA/CUDNN/CUBLAS calls
│   └── scheduler                 # Implementation of the scheduling policy
│   └── scheduler_frontend.py     # Python interface for the Orion scheduler
└── benchmarking                  # Scripts and configuration files for benchmarking
|   ├── benchmark_suite           # Training and inference scripts
|   ├── model_kernels             # Files containing profile information for the submitted models
└── related                       # Some of the related baselines: MPS, Streams, Tick-Tock
└── artifact_evaluation           # Scripts and instructions for artifact evaluation
|   ├── example                   # Basic example to test Orion functionality
|   ├── FICKS                     # FICKS
|   ├── fig7                      # Scripts to reproduce Figure 7 of the paper
|   ├── fig10                     # Scripts to reproduce Figure 10 of the paper
└── setup                         # Instructions and scripts to install Orion's prerequisites.
```

## For code review

### Scheduler: 
The scheduler is located at src/scheduler/scheduler_eval.cpp, the sheduelr function is named busy_wait_ficks, with some debug prints.

### Spatial Sharing: 
spatial sharing is implemented using libsmctrl, and we use set_mask and unset_mask in the scheduler to specify the TPC for running the kernel.


### Inference execution:
run run_orion.py in artifact_evaluation/FICKS/, we can specify the the combination of the model and batch size for model Mnet(mobilenetv2) and Rnet(Resnet152), and the parameters.


