#pragma once
int hist_with_cuda(const int* A, int* results, int size, int threadsPerBlock, int blocksPerGrid);
