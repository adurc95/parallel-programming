#include <cuda_runtime.h>
#include <stdio.h>
#include "cudaFuncs.h"
#include "constants.h"

__global__ void hist_kernel(const int *data, int *histogram, int size) {
    extern __shared__ int sharedHistogram[];

    // Initialize shared memory
    for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
        sharedHistogram[i] = 0;
    }
    __syncthreads();

    // Compute histogram in shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(&(sharedHistogram[data[tid]]), 1);
    }
    __syncthreads();

    // Accumulate histogram in global memory
    for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
        atomicAdd(&(histogram[i]), sharedHistogram[i]);
    }
}

int hist_with_cuda(const int *data, int *histogram, int size, int threadsPerBlock, int blocksPerGrid) {
    int *dev_data, *dev_histogram;

    // Allocate GPU memory
    cudaMalloc((void **)&dev_data, size * sizeof(int));
    cudaMalloc((void **)&dev_histogram, HISTOGRAM_SIZE * sizeof(int));

    // Copy input data to GPU
    cudaMemcpy(dev_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize histogram on GPU
    cudaMemset(dev_histogram, 0, HISTOGRAM_SIZE * sizeof(int));

    // Launch kernel to compute histogram
    hist_kernel<<<blocksPerGrid, threadsPerBlock, HISTOGRAM_SIZE * sizeof(int)>>>(dev_data, dev_histogram, size);

    // Copy histogram back to host
    cudaMemcpy(histogram, dev_histogram, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up GPU memory
    cudaFree(dev_data);
    cudaFree(dev_histogram);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}

