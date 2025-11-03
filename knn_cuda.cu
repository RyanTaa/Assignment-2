#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cfloat>
#include <cuda_runtime.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include "libarff/arff_attr.h"
#include "libarff/arff_value.h"

__host__ float cpu_distance(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;
    for (int i = 0; i < num_attributes - 1; i++) {
        float diff = instance_A[i] - instance_B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

__host__ int* serialKNN(float* train_matrix, float* test_matrix,
                       int num_classes, int num_attributes,
                       int train_num_instances, int test_num_instances, int k) 
{    
    int* predictions = (int*)malloc(test_num_instances * sizeof(int));
    float* candidates = (float*) calloc(k * 2, sizeof(float));
    int* classCounts = (int*)calloc(num_classes, sizeof(int));

    for(int queryIndex = 0; queryIndex < test_num_instances; queryIndex++) {
        for(int i = 0; i < 2 * k; i++){ candidates[i] = FLT_MAX; }
        memset(classCounts, 0, num_classes * sizeof(int));

        for(int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
            float dist = cpu_distance(&test_matrix[queryIndex * num_attributes], 
                                    &train_matrix[keyIndex * num_attributes], 
                                    num_attributes);
            
            for(int c = 0; c < k; c++){
                if(dist < candidates[2 * c]) {
                    for(int x = k - 2; x >= c; x--) {
                        candidates[2 * x + 2] = candidates[2 * x];
                        candidates[2 * x + 3] = candidates[2 * x + 1];
                    }
                    candidates[2 * c] = dist;
                    candidates[2 * c + 1] = train_matrix[keyIndex * num_attributes + num_attributes - 1]; // class value
                    break;
                }
            }
        }

        for(int i = 0; i < k; i++) {
            // Check for valid class index before incrementing
            int classIdx = (int)candidates[2 * i + 1];
            if(classIdx >= 0 && classIdx < num_classes) {
                classCounts[classIdx]++;
            }
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

// Calculates distance
__device__ float gpu_distance(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;
    for (int i = 0; i < num_attributes - 1; i++) {
        float diff = instance_A[i] - instance_B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}


// Naive Implementation
__global__ void knn_naive_kernel(float* d_train_matrix, float* d_test_matrix, int* d_predictions,
                                 int k, int num_classes, int num_attributes,
                                 int train_num_instances, int test_num_instances)
{
    // Map thread to a test instance
    int queryIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryIndex >= test_num_instances) {
        return;
    }

    // Allocate private candidate and class count arrays in heap
    float* candidates = (float*) malloc(k * 2 * sizeof(float));
    int* classCounts = (int*) malloc(num_classes * sizeof(int));
    if (candidates == NULL || classCounts == NULL) {
        return; 
    }

    for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }
    for (int i = 0; i < num_classes; i++) { classCounts[i] = 0; }

    float* test_instance = &d_test_matrix[queryIndex * num_attributes];

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

    for (int i = 0; i < k; i++) {
        int classIdx = (int)candidates[2 * i + 1];
        if(classIdx >= 0 && classIdx < num_classes) {
            classCounts[classIdx]++;
        }
    }

    int max_value = -1;
    int max_class = 0;
    for (int i = 0; i < num_classes; i++) {
        if (classCounts[i] > max_value) {
            max_value = classCounts[i];
            max_class = i;
        }
    }
    d_predictions[queryIndex] = max_class;

    free(candidates);
    free(classCounts);
}


// Shared Memory Implementation
#define K_MAX 100         
#define NUM_CLASSES_MAX 50  
#define MAX_ATTR 128      

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
    __shared__ float s_train_instance[MAX_ATTR];

    int queryIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryIndex >= test_num_instances) {
        return;
    }

    // Check hard-coded limits
    if(k > K_MAX || num_classes > NUM_CLASSES_MAX || num_attributes > MAX_ATTR) {
        return;
    }

    float candidates[K_MAX * 2];
    int classCounts[NUM_CLASSES_MAX];
    float my_test_instance[MAX_ATTR]; 

    for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }
    for (int i = 0; i < num_classes; i++) { classCounts[i] = 0; }

    for (int i = 0; i < num_attributes; i++) {
        my_test_instance[i] = d_test_matrix[queryIndex * num_attributes + i];
    }

    for (int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
        
        if (threadIdx.x < num_attributes) {
            s_train_instance[threadIdx.x] = d_train_matrix[keyIndex * num_attributes + threadIdx.x];
        }
        __syncthreads();

        float dist = gpu_distance_shared(my_test_instance, s_train_instance, num_attributes);

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
        __syncthreads();
    }

    for (int i = 0; i < k; i++) {
        int classIdx = (int)candidates[2 * i + 1];
        if(classIdx >= 0 && classIdx < num_classes) {
            classCounts[classIdx]++;
        }
    }

    int max_value = -1;
    int max_class = 0;
    for (int i = 0; i < num_classes; i++) {
        if (classCounts[i] > max_value) {
            max_value = classCounts[i];
            max_class = i;
        }
    }
    d_predictions[queryIndex] = max_class;
}

// Builds the flat float* matrix from the ArffData object.
float* build_dataset_matrix(ArffData* data) {
    int num_attrs = data->num_attributes();
    int num_inst = data->num_instances();
    long long num_elements = (long long)num_inst * num_attrs;
    float* matrix = (float*)malloc(num_elements * sizeof(float));
    if (matrix == NULL) {
        fprintf(stderr, "Failed to allocate memory for dataset matrix!\n");
        exit(EXIT_FAILURE);
    }

    ArffAttr* class_attr = data->get_attr(num_attrs - 1);
    ArffNominal nominals;
    if (class_attr->type() == NOMINAL) {
        nominals = data->get_nominal(class_attr->name());
    }

    for(int i = 0; i < num_inst; i++) {
        ArffInstance* inst = data->get_instance(i);
        for(int j = 0; j < num_attrs; j++) {
            ArffValue* val = inst->get(j);
            float float_val = 0.0;

            if (j == num_attrs - 1) { // Class attribute
                if (class_attr->type() == NOMINAL) {
                    if (val->missing()) {
                        float_val = -1.0; // Placeholder for missing class
                    } else {
                        std::string str_val = (std::string)*val;
                        bool found = false;
                        for(size_t n = 0; n < nominals.size(); n++) {
                            if(nominals[n] == str_val) {
                                float_val = (float)n;
                                found = true;
                                break;
                            }
                        }
                        if (!found) float_val = -1.0; // Error
                    }
                } else {
                     float_val = (val->missing()) ? 0.0 : (float)*val;
                }
            } else { // Regular attribute
                float_val = (val->missing()) ? 0.0 : (float)*val;
            }
            matrix[i * num_attrs + j] = float_val;
        }
    }
    return matrix;
}


int* computeConfusionMatrix(int* predictions, ArffData* dataset, int num_classes)
{    
    int* confusionMatrix = (int*)calloc(num_classes * num_classes, sizeof(int));
    ArffAttr* class_attr = dataset->get_attr(dataset->num_attributes() - 1); 
    
    ArffNominal nominals; // Declare outside loop
    bool is_nominal = (class_attr->type() == NOMINAL);
    if(is_nominal) {
        nominals = dataset->get_nominal(class_attr->name());
    }

    for(int i = 0; i < dataset->num_instances(); i++) { 
        ArffValue* trueVal = dataset->get_instance(i)->get(dataset->num_attributes() - 1);
        
        int trueClass = -1;
        if(trueVal->missing()) {
             fprintf(stderr, "Warning: Missing true class value in test set at index %d!\n", i);
             continue;
        }

        if (is_nominal) {
            std::string str_val = (std::string)*trueVal;
            for(size_t n = 0; n < nominals.size(); n++) {
                if(nominals[n] == str_val) {
                    trueClass = (int)n;
                    break;
                }
            }
            if(trueClass == -1) {
                 fprintf(stderr, "Error: Unknown nominal class value '%s' in test set!\n", ((std::string)*trueVal).c_str());
                 continue;
            }
        } else {
            trueClass = (int)((float)*trueVal);        
        }

        int predictedClass = predictions[i];
        
        if (predictedClass >= 0 && predictedClass < num_classes &&
            trueClass >= 0 && trueClass < num_classes) 
        {
            confusionMatrix[trueClass * num_classes + predictedClass]++;
        }
    }
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset, int num_classes)
{
    int successfulPredictions = 0;
    for(int i = 0; i < num_classes; i++) {
        successfulPredictions += confusionMatrix[i * num_classes + i];
    }
    return 100 * successfulPredictions / (float) dataset->num_instances();
}

void verifyResults(int* serialPredictions, int* gpuPredictions, int num_instances) {
    for (int i = 0; i < num_instances; i++) {
        if (serialPredictions[i] != gpuPredictions[i]) {
            fprintf(stderr, "Error: Mismatch at index %d! CPU: %d, GPU: %d\n", 
                    i, serialPredictions[i], gpuPredictions[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("CPU and GPU results match.\n");
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

    printf("Parsing training file: %s\n", argv[1]);
    ArffParser parserTrain(argv[1]);
    ArffData *train = parserTrain.parse();
    printf("Parsing test file: %s\n", argv[2]);
    ArffParser parserTest(argv[2]);
    ArffData *test = parserTest.parse();
    
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    ArffAttr* class_attr = train->get_attr(num_attributes - 1);
    int num_classes = 0; 

    if (class_attr->type() == NOMINAL) {
        ArffNominal nominals = train->get_nominal(class_attr->name());
        num_classes = nominals.size();
    } 
    else if (class_attr->type() == NUMERIC)    
    {
        printf("Warning: Last attribute '%s' is not nominal. Assuming numeric class labels (0, 1, 2, ...).\n", class_attr->name().c_str());
        printf("Finding max class index from training data...\n");
        float max_class_val = -1.0;
        
        // Iterate over ArffData object to find the max class index
        for (int i = 0; i < train_num_instances; i++) {
            ArffInstance* inst = train->get_instance(i);
            ArffValue* val = inst->get(num_attributes - 1); // Get class value
            
            if (!val->missing()) {
                float class_val = (float)*val;
                if (class_val > max_class_val) {
                    max_class_val = class_val;
                }
            }
        }

        if (max_class_val < 0) {
            fprintf(stderr, "Error: Could not find any valid positive numeric class labels in training data.\n");
            exit(1);
        }
        num_classes = (int)max_class_val + 1; 
        printf("Max class index found: %d. Setting num_classes = %d\n", (int)max_class_val, num_classes);

    } 
    else {
        fprintf(stderr, "Error: Last attribute '%s' is not NOMINAL or NUMERIC/REAL!\n", class_attr->name().c_str());
        exit(1);
    }


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
    bool can_run_shared = true;
    if (k > K_MAX || num_classes > NUM_CLASSES_MAX || num_attributes > MAX_ATTR) {
        printf("Warning: Problem size exceeds limits for optimized shared kernel.\n");
        printf("k (max %d): %d, classes (max %d): %d, attributes (max %d): %d\n",
                K_MAX, k, NUM_CLASSES_MAX, num_classes, MAX_ATTR, num_attributes);
        printf("Only running Naive and Serial versions.\n");
        can_run_shared = false;
    }

    printf("Building dataset matrices from ARFF data...\n");
    float* h_train_matrix = build_dataset_matrix(train);
    float* h_test_matrix = build_dataset_matrix(test);
    printf("Matrix build complete.\n\n");

    int* h_predictions_serial = NULL;
    int* h_predictions_gpu = (int*)malloc(predictions_size);

    float *d_train_matrix, *d_test_matrix;
    int *d_predictions;
    CUDA_CHECK(cudaMalloc(&d_train_matrix, train_data_size));
    CUDA_CHECK(cudaMalloc(&d_test_matrix, test_data_size));
    CUDA_CHECK(cudaMalloc(&d_predictions, predictions_size));

    // Transfer Data to Device (GPU)
    CUDA_CHECK(cudaMemcpy(d_train_matrix, h_train_matrix, train_data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test_matrix, h_test_matrix, test_data_size, cudaMemcpyHostToDevice));

    // CUDA Events for Timing
    cudaEvent_t start, stop;
    float ms_serial = 0, ms_gpu_naive = 0, ms_gpu_shared = 0;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    // Run CPU Serial Version
    printf("Running Serial CPU KNN...\n");
    cudaEventRecord(start, 0);
    h_predictions_serial = serialKNN(h_train_matrix, h_test_matrix,
                                     num_classes, num_attributes,
                                     train_num_instances, test_num_instances, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_serial, start, stop);

    int* confusionMatrix_serial = computeConfusionMatrix(h_predictions_serial, test, num_classes);
    float accuracy_serial = computeAccuracy(confusionMatrix_serial, test, num_classes);    printf("Serial CPU Time: %.2f ms\n", ms_serial);
    printf("Serial CPU Accuracy: %.2f%%\n\n", accuracy_serial);
    
    // Run CUDA Naive Kernel
    printf("Running CUDA Naive Kernel...\n");
    dim3 gridDim((test_num_instances + threads_per_block - 1) / threads_per_block, 1, 1);
    dim3 blockDim(threads_per_block, 1, 1);
    
    CUDA_CHECK(cudaMemset(d_predictions, 0, predictions_size));

    cudaEventRecord(start, 0);
    knn_naive_kernel<<<gridDim, blockDim>>>(d_train_matrix, d_test_matrix, d_predictions,
                                             k, num_classes, num_attributes,
                                             train_num_instances, test_num_instances);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_gpu_naive, start, stop);
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_predictions_gpu, d_predictions, predictions_size, cudaMemcpyDeviceToHost));
    
    // Verify and print results
    verifyResults(h_predictions_serial, h_predictions_gpu, test_num_instances);
    printf("CUDA Naive Time: %.2f ms\n", ms_gpu_naive);
    printf("CUDA Naive Speedup vs Serial: %.2f x\n\n", ms_serial / ms_gpu_naive);

    // 3. Run CUDA Shared Memory Kernel 
    if (can_run_shared) {
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
    if (ms_gpu_shared > 0 && (!can_run_shared || ms_gpu_shared < ms_gpu_naive) ) { // Only show shared if it ran and was faster
        printf("CUDA Shared Time:     %10.2f ms   (Speedup: %.2fx)\n", ms_gpu_shared, ms_serial / ms_gpu_shared);
    }
    printf("---------------------\n");

    // --- Cleanup ---
    free(h_train_matrix);
    free(h_test_matrix);
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