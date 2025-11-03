// knn_cuda.cu
//
// Implements a naive and a shared-memory-optimized CUDA kernel for KNN.
// To compile:
// nvcc -O3 -arch=sm_70 -o knn_cuda knn_cuda.cu libarff/libarff.a -lm
// (Adjust -arch=sm_70 for the GPU architecture you are using, e.g., sm_60 for P100, sm_70 for V100, sm_80 for A100)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cfloat>
#include <cuda_runtime.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

//
// --- CPU Serial Code (for baseline) ---
// (Copied from serial.cpp)
//

__host__ float cpu_distance(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;
    for (int i = 0; i < num_attributes - 1; i++) {
        float diff = instance_A[i] - instance_B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

__host__ int* serialKNN(ArffData* train, ArffData* test, int k) {    
    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    int* predictions = (int*)malloc(test_num_instances * sizeof(int));
    float* candidates = (float*) calloc(k * 2, sizeof(float));
    float* train_matrix = train->get_dataset_matrix();
    float* test_matrix = test->get_dataset_matrix();
    int* classCounts = (int*)calloc(num_classes, sizeof(int));

    for(int queryIndex = 0; queryIndex < test_num_instances; queryIndex++) {
        for(int i = 0; i < 2 * k; i++){ candidates[i] = FLT_MAX; }
        memset(classCounts, 0, num_classes * sizeof(int));

        for(int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
            float dist = cpu_distance(&test_matrix[queryIndex * num_attributes], &train_matrix[keyIndex * num_attributes], num_attributes);
            for(int c = 0; c < k; c++){
                if(dist < candidates[2 * c]) {
                    for(int x = k - 2; x >= c; x--) {
                        candidates[2 * x + 2] = candidates[2 * x];
                        candidates[2 * x + 3] = candidates[2 * x + 1];
                    }
                    candidates[2 * c] = dist;
                    candidates[2 * c + 1] = train_matrix[keyIndex * num_attributes + num_attributes - 1];
                    break;
                }
            }
        }

        for(int i = 0; i < k; i++) {
            classCounts[(int)candidates[2 * i + 1]]++;
        }
        
        int max_value = -1;
        int max_class = 0;
        for(int i = 0; i < num_classes; i++) {
            if(classCounts[i] > max_value) {
                max_value = classCounts[i];
                max_class = i;
            }
        }
        predictions[queryIndex] = max_class;
    }

    free(candidates);
    free(classCounts);
    return predictions;
}

//
// --- CUDA Helper Code ---
//

// Macro for checking CUDA errors
#define CUDA_CHECK(err) { \
    cudaError_t error = err; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

// Device function to calculate distance
__device__ float gpu_distance(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;
    for (int i = 0; i < num_attributes - 1; i++) {
        float diff = instance_A[i] - instance_B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

//
// --- CUDA Kernel 1: Naive Implementation ---
//
// Each thread handles one test instance.
// All data is read from global memory.
// Uses malloc/free inside the kernel (slow).
//
__global__ void knn_naive_kernel(float* d_train_matrix, float* d_test_matrix, int* d_predictions,
                                 int k, int num_classes, int num_attributes,
                                 int train_num_instances, int test_num_instances)
{
    // Map thread to a test instance
    int queryIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryIndex >= test_num_instances) {
        return;
    }

    // Allocate private candidate and class count arrays in heap (slow)
    // This directly mirrors the logic from threaded.cpp's knn_thread_func
    float* candidates = (float*) malloc(k * 2 * sizeof(float));
    int* classCounts = (int*) malloc(num_classes * sizeof(int));
    if (candidates == NULL || classCounts == NULL) {
        // Handle allocation failure if necessary
        return; 
    }

    for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }
    for (int i = 0; i < num_classes; i++) { classCounts[i] = 0; }

    // Get pointer to this thread's test instance
    float* test_instance = &d_test_matrix[queryIndex * num_attributes];

    // Find k-nearest neighbors
    for (int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
        float* train_instance = &d_train_matrix[keyIndex * num_attributes];
        float dist = gpu_distance(test_instance, train_instance, num_attributes);

        for (int c = 0; c < k; c++) {
            if (dist < candidates[2 * c]) {
                for (int x = k - 2; x >= c; x--) {
                    candidates[2 * x + 2] = candidates[2 * x];
                    candidates[2 * x + 3] = candidates[2 * x + 1];
                }
                candidates[2 * c] = dist;
                candidates[2 * c + 1] = train_instance[num_attributes - 1]; // Class label
                break;
            }
        }
    }

    // Bincount the results
    for (int i = 0; i < k; i++) {
        classCounts[(int)candidates[2 * i + 1]]++;
    }

    // Find the max class
    int max_value = -1;
    int max_class = 0;
    for (int i = 0; i < num_classes; i++) {
        if (classCounts[i] > max_value) {
            max_value = classCounts[i];
            max_class = i;
        }
    }

    // Write prediction to global memory
    d_predictions[queryIndex] = max_class;

    // Free heap memory
    free(candidates);
    free(classCounts);
}


//
// --- CUDA Kernel 2: Shared Memory Implementation ---
//
// Hard-coded limits for static arrays.
// This avoids in-kernel malloc/free, which is much faster.
#define K_MAX 100         // Max k value allowed
#define NUM_CLASSES_MAX 50  // Max number of classes
#define MAX_ATTR 128      // Max number of attributes

// Device function for distance calculation using private and shared memory
__device__ float gpu_distance_shared(float* private_test_instance, float* shared_train_instance, int num_attributes) {
    float sum = 0;
    for (int i = 0; i < num_attributes - 1; i++) {
        float diff = private_test_instance[i] - shared_train_instance[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

__global__ void knn_shared_kernel(float* d_train_matrix, float* d_test_matrix, int* d_predictions,
                                  int k, int num_classes, int num_attributes,
                                  int train_num_instances, int test_num_instances)
{
    // Each block loads one train instance at a time into shared memory.
    // Each thread in the block computes its distance against that shared instance.
    
    // Declare shared memory for one training instance
    __shared__ float s_train_instance[MAX_ATTR];

    // Map thread to a test instance
    int queryIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryIndex >= test_num_instances) {
        return;
    }

    // Use fast static local arrays instead of malloc/free
    float candidates[K_MAX * 2];
    int classCounts[NUM_CLASSES_MAX];
    float my_test_instance[MAX_ATTR]; // Load test instance into private registers

    // Initialize private arrays
    for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }
    for (int i = 0; i < num_classes; i++) { classCounts[i] = 0; }

    // Load this thread's test instance into its private memory (registers)
    // This is one read from global memory per thread.
    for (int i = 0; i < num_attributes; i++) {
        my_test_instance[i] = d_test_matrix[queryIndex * num_attributes + i];
    }

    // Loop over all training instances
    for (int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
        
        // --- Cooperative Load ---
        // Have the first 'num_attributes' threads in the block load
        // the current training instance into shared memory.
        if (threadIdx.x < num_attributes) {
            s_train_instance[threadIdx.x] = d_train_matrix[keyIndex * num_attributes + threadIdx.x];
        }
        // Wait for all threads to finish loading
        __syncthreads();

        // --- Compute Distance ---
        // Now, all threads in the block compute their distance against the
        // *same* training instance, reading from fast shared memory.
        float dist = gpu_distance_shared(my_test_instance, s_train_instance, num_attributes);

        // --- Update Candidates (in private memory) ---
        for (int c = 0; c < k; c++) {
            if (dist < candidates[2 * c]) {
                for (int x = k - 2; x >= c; x--) {
                    candidates[2 * x + 2] = candidates[2 * x];
                    candidates[2 * x + 3] = candidates[2 * x + 1];
                }
                candidates[2 * c] = dist;
                candidates[2 * c + 1] = s_train_instance[num_attributes - 1]; // Class label
                break;
            }
        }
        
        // Wait for all threads to finish using the shared instance
        // before loading the next one.
        __syncthreads();
    }

    // --- Bincount (private) ---
    for (int i = 0; i < k; i++) {
        classCounts[(int)candidates[2 * i + 1]]++;
    }

    // --- Find Max Class (private) ---
    int max_value = -1;
    int max_class = 0;
    for (int i = 0; i < num_classes; i++) {
        if (classCounts[i] > max_value) {
            max_value = classCounts[i];
            max_class = i;
        }
    }

    // Write final prediction to global memory
    d_predictions[queryIndex] = max_class;
}


//
// --- Host Main Function ---
//

// Helper functions from serial.cpp
int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int));
    for(int i = 0; i < dataset->num_instances(); i++) {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        confusionMatrix[trueClass * dataset->num_classes() + predictedClass]++;
    }
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    for(int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i * dataset->num_classes() + i];
    }
    return 100 * successfulPredictions / (float) dataset->num_instances();
}

// Function to check results
void verifyResults(int* serialPredictions, int* gpuPredictions, int num_instances) {
    for (int i = 0; i < num_instances; i++) {
        if (serialPredictions[i] != gpuPredictions[i]) {
            fprintf(stderr, "Error: Mismatch at index %d! CPU: %d, GPU: %d\n", 
                    i, serialPredictions[i], gpuPredictions[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Verification: SUCCESS! CPU and GPU results match.\n");
}


int main(int argc, char *argv[])
{
    if(argc != 5)
    {
        printf("Usage: ./knn_cuda datasets/train.arff datasets/test.arff k threads_per_block\n");
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);
    int threads_per_block = strtol(argv[4], NULL, 10);

    // --- Load Data ---
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    
    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();
    long long train_data_size = (long long)train_num_instances * num_attributes * sizeof(float);
    long long test_data_size = (long long)test_num_instances * num_attributes * sizeof(float);
    long long predictions_size = (long long)test_num_instances * sizeof(int);

    printf("--- KNN Problem Setup ---\n");
    printf("Train instances: %d\n", train_num_instances);
    printf("Test instances:  %d\n", test_num_instances);
    printf("Attributes:      %d\n", num_attributes);
    printf("Classes:         %d\n", num_classes);
    printf("K value:         %d\n", k);
    printf("CUDA Block Size: %d\n", threads_per_block);
    printf("-------------------------\n\n");

    // Check for optimized kernel limits
    if (k > K_MAX || num_classes > NUM_CLASSES_MAX || num_attributes > MAX_ATTR) {
        printf("Warning: Problem size exceeds limits for optimized shared kernel.\n");
        printf("k (max %d): %d, classes (max %d): %d, attributes (max %d): %d\n",
                K_MAX, k, NUM_CLASSES_MAX, num_classes, MAX_ATTR, num_attributes);
        printf("Only running Naive and Serial versions.\n");
    }

    // --- Host Memory Allocation ---
    float* h_train_matrix = train->get_dataset_matrix();
    float* h_test_matrix = test->get_dataset_matrix();
    int* h_predictions_serial = NULL;
    int* h_predictions_gpu = (int*)malloc(predictions_size);

    // --- Device Memory Allocation ---
    float *d_train_matrix, *d_test_matrix;
    int *d_predictions;
    CUDA_CHECK(cudaMalloc(&d_train_matrix, train_data_size));
    CUDA_CHECK(cudaMalloc(&d_test_matrix, test_data_size));
    CUDA_CHECK(cudaMalloc(&d_predictions, predictions_size));

    // --- Transfer Data to Device (GPU) ---
    CUDA_CHECK(cudaMemcpy(d_train_matrix, h_train_matrix, train_data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test_matrix, h_test_matrix, test_data_size, cudaMemcpyHostToDevice));

    // --- CUDA Events for Timing ---
    cudaEvent_t start, stop;
    float ms_serial = 0, ms_gpu_naive = 0, ms_gpu_shared = 0;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    //
    // --- 1. Run CPU Serial Version (Baseline) ---
    //
    printf("Running Serial CPU KNN...\n");
    cudaEventRecord(start, 0);
    h_predictions_serial = serialKNN(train, test, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_serial, start, stop);

    int* confusionMatrix_serial = computeConfusionMatrix(h_predictions_serial, test);
    float accuracy_serial = computeAccuracy(confusionMatrix_serial, test);
    printf("Serial CPU Time: %.2f ms\n", ms_serial);
    printf("Serial CPU Accuracy: %.2f%%\n\n", accuracy_serial);
    
    //
    // --- 2. Run CUDA Naive Kernel ---
    //
    printf("Running CUDA Naive Kernel...\n");
    dim3 gridDim((test_num_instances + threads_per_block - 1) / threads_per_block, 1, 1);
    dim3 blockDim(threads_per_block, 1, 1);
    
    CUDA_CHECK(cudaMemset(d_predictions, 0, predictions_size)); // Clear prediction buffer

    cudaEventRecord(start, 0);
    knn_naive_kernel<<<gridDim, blockDim>>>(d_train_matrix, d_test_matrix, d_predictions,
                                             k, num_classes, num_attributes,
                                             train_num_instances, test_num_instances);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_gpu_naive, start, stop);
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_predictions_gpu, d_predictions, predictions_size, cudaMemcpyDeviceToHost));
    
    // Verify and print results
    verifyResults(h_predictions_serial, h_predictions_gpu, test_num_instances);
    printf("CUDA Naive Time: %.2f ms\n", ms_gpu_naive);
    printf("CUDA Naive Speedup vs Serial: %.2f x\n\n", ms_serial / ms_gpu_naive);

    //
    // --- 3. Run CUDA Shared Memory Kernel ---
    //
    if (k <= K_MAX && num_classes <= NUM_CLASSES_MAX && num_attributes <= MAX_ATTR) {
        printf("Running CUDA Shared Memory Kernel...\n");
        CUDA_CHECK(cudaMemset(d_predictions, 0, predictions_size)); // Clear buffer

        cudaEventRecord(start, 0);
        knn_shared_kernel<<<gridDim, blockDim>>>(d_train_matrix, d_test_matrix, d_predictions,
                                                k, num_classes, num_attributes,
                                                train_num_instances, test_num_instances);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_gpu_shared, start, stop);

        // Copy results back
        CUDA_CHECK(cudaMemcpy(h_predictions_gpu, d_predictions, predictions_size, cudaMemcpyDeviceToHost));
        
        // Verify and print results
        verifyResults(h_predictions_serial, h_predictions_gpu, test_num_instances);
        printf("CUDA Shared Time: %.2f ms\n", ms_gpu_shared);
        printf("CUDA Shared Speedup vs Serial: %.2f x\n", ms_serial / ms_gpu_shared);
        printf("CUDA Shared Speedup vs Naive:  %.2f x\n\n", ms_gpu_naive / ms_gpu_shared);
    }

    // --- Final Summary ---
    printf("--- Final Results ---\n");
    printf("Serial CPU Time:      %10.2f ms   (Accuracy: %.2f%%)\n", ms_serial, accuracy_serial);
    printf("CUDA Naive Time:      %10.2f ms   (Speedup: %.2fx)\n", ms_gpu_naive, ms_serial / ms_gpu_naive);
    if (ms_gpu_shared > 0) {
        printf("CUDA Shared Time:     %10.2f ms   (Speedup: %.2fx)\n", ms_gpu_shared, ms_serial / ms_gpu_shared);
    }
    printf("---------------------\n");

    // --- Cleanup ---
    free(h_predictions_serial);
    free(h_predictions_gpu);
    free(confusionMatrix_serial);
    delete train;
    delete test;
    CUDA_CHECK(cudaFree(d_train_matrix));
    CUDA_CHECK(cudaFree(d_test_matrix));
    CUDA_CHECK(cudaFree(d_predictions));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}