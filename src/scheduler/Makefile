# Compiler and flags
CC=g++
NVCC=/usr/local/cuda-12.3/bin/nvcc
CFLAGS=-O3 -std=c++17

CUDAINCLUDE=/usr/local/cuda-12.3/include/
CUDALIB=/usr/local/cuda-12.3/targets/x86_64-linux/lib

# ---- cuDNN from conda env (same one PyTorch uses) ----
CUDNN_INCDIR=/home/zixi/miniconda3/envs/orion/lib/python3.8/site-packages/nvidia/cudnn/include
CUDNN_LIBDIR=/home/zixi/miniconda3/envs/orion/lib/python3.8/site-packages/nvidia/cudnn/lib
CUDNN_SO=$(CUDNN_LIBDIR)/libcudnn.so.9

SMCTRL_CFLAGS=-fPIC
SMCTRL_LDFLAGS=-lcuda -I/usr/local/cuda/include -ldl

DEBUG_FLAGS=
# DEBUG_FLAGS=-DDEBUG

# GPU architectures (A4000 = sm_86, RTX 6000 Ada = sm_89)
GENCODE_FLAGS= \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89

NVCCFLAGS=$(CFLAGS) $(DEBUG_FLAGS) -Xcompiler -fPIC $(GENCODE_FLAGS) -Wno-deprecated-gpu-targets

OBJ_FILES=utils_sched.o scheduler_eval.o libsmctrl.o

all: scheduler_eval.so

libsmctrl.o: libsmctrl.cpp libsmctrl.h
	$(CC) $(SMCTRL_CFLAGS) $(DEBUG_FLAGS) -c libsmctrl.cpp -I$(CUDAINCLUDE) -o libsmctrl.o

utils_sched.o: utils_sched.cpp
	$(NVCC) $(NVCCFLAGS) -c utils_sched.cpp -I$(CUDAINCLUDE) -I$(CUDNN_INCDIR)

scheduler_eval.o: scheduler_eval.cpp
	$(NVCC) $(NVCCFLAGS) -c scheduler_eval.cpp -I$(CUDAINCLUDE) -I$(CUDNN_INCDIR)

scheduler_eval.so: $(OBJ_FILES)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -fPIC -shared $(OBJ_FILES) \
		-I$(CUDAINCLUDE) -I$(CUDNN_INCDIR) \
		-L$(CUDALIB) \
		-Wl,-rpath,$(CUDNN_LIBDIR) \
		-lcudart -lcuda -lcublas \
		$(CUDNN_SO) \
		-lpthread -lboost_thread -lboost_system \
		-o scheduler_eval.so

clean:
	rm -rf *.o *.so
