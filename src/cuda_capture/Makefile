CC=g++
CUDAINCLUDE=/usr/local/cuda-12.3/include/
CUDALIB=/usr/local/cuda-12.3/targets/x86_64-linux/lib

# ---- cuDNN from conda env (same one PyTorch uses) ----
CUDNN_INCDIR=/home/zixi/miniconda3/envs/orion/lib/python3.8/site-packages/nvidia/cudnn/include
CUDNN_LIBDIR=/home/zixi/miniconda3/envs/orion/lib/python3.8/site-packages/nvidia/cudnn/lib
CUDNN_SO=$(CUDNN_LIBDIR)/libcudnn.so.9

DEBUG_FLAGS=
# DEBUG_FLAGS=-DDEBUG

libinttemp.so: utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp
	$(CC) $(DEBUG_FLAGS) -O3 -std=c++17 -fPIC -shared \
		utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp \
		-I$(CUDAINCLUDE) -I$(CUDNN_INCDIR) \
		-L$(CUDALIB) \
		-Wl,-rpath,$(CUDNN_LIBDIR) \
		-lcudart -lcublas \
		$(CUDNN_SO) \
		-lpthread \
		-o libinttemp.so

all: libinttemp.so

clean:
	rm -rf *.o libinttemp.so
