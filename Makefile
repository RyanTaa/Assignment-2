# This Makefile uses an explicit, simple style.
# It compiles all .cpp/.cu files for each target in one line.

all: serial threaded mpi knn_cuda

serial: serial.cpp libarff/arff_parser.cpp libarff/arff_data.cpp
	g++ -O3 -Wall -Ilibarff -o serial serial.cpp libarff/arff_parser.cpp libarff/arff_data.cpp -lrt -lm

threaded: threaded.cpp libarff/arff_parser.cpp libarff/arff_data.cpp
	g++ -O3 -Wall -Ilibarff -o threaded threaded.cpp libarff/arff_parser.cpp libarff/arff_data.cpp -lrt -lm -lpthread

mpi: mpi.cpp libarff/arff_parser.cpp libarff/arff_data.cpp
	mpic++ -O3 -Wall -Ilibarff -o mpi mpi.cpp libarff/arff_parser.cpp libarff/arff_data.cpp -lrt -lm

knn_cuda: knn_cuda.cu libarff/arff_parser.cpp libarff/arff_data.cpp
	# nvcc can compile both .cu and .cpp files.
	# Make sure -arch=sm_70 matches your GPU (sm_70 for V100, sm_60 for P100)
	nvcc -O3 -arch=sm_70 -Ilibarff -o knn_cuda knn_cuda.cu libarff/arff_parser.cpp libarff/arff_data.cpp -lrt -lm

clean:
	rm -f serial threaded mpi knn_cuda