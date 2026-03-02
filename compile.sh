#!/bin/bash
# Define directories
CUDA_CAPTURE_DIR="./src/cuda_capture/"
SCHEDULER_DIR="./src/scheduler/"

# Clean directories
echo "Cleaning directory: $CUDA_CAPTURE_DIR"
cd $CUDA_CAPTURE_DIR && make clean

cd ../..

echo "Cleaning directory: $SCHEDULER_DIR"
cd $SCHEDULER_DIR && make clean

# Go back to the root directory (if necessary)
cd ../..

cd src/cuda_capture && make libinttemp.so && cd ../../
cd src/scheduler && make scheduler_eval.so && cd ../../
