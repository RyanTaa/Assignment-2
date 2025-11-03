# This Makefile uses a variable to automatically find all .cpp files
# in the libarff directory, so they don't need to be listed manually.

# Automatically find all .cpp files in the libarff directory
LIBARFF_SRCS = $(wildcard libarff/*.cpp)

knn_cuda: knn_cuda.cu
	nvcc -O3 -arch=sm_60 -Ilibarff -o knn_cuda knn_cuda.cu $(LIBARFF_SRCS) -lrt -lm

clean:
	rm -f knn_cuda