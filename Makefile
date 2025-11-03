# This Makefile uses a variable to automatically find all .cpp files
# in the libarff directory, so they don't need to be listed manually.

# Automatically find all .cpp files in the libarff directory
LIBARFF_SRCS = $(wildcard libarff/*.cpp)

knn_cuda: knn_cuda.cu
	# nvcc can compile both .cu and .cpp files.
	# This command now correctly includes $(LIBARFF_SRCS)
	# Make sure -arch=sm_70 matches your GPU (sm_70 for V100, sm_60 for P100)
	nvcc -O3 -arch=sm_70 -Ilibarff -o knn_cuda knn_cuda.cu $(LIBARFF_SRCS) -lrt -lm

clean:
	rm -f serial threaded mpi knn_cuda