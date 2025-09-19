#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> 
#include <omp.h>
#include <cstring>
#include "mpi.h"
#include "constants.h"
#include "cudaFuncs.h"


void initialize_mpi(int *argc, char ***argv, int *size, int *rank);
int* generate_random_array(int data_size);
void divide_and_send(int *data, int data_size, int part, int rank);
void receive_data(int **data, int *part, int rank);
void process_data(int *data, int part, int rank);
void finalize_mpi(int *data, int *result, int *omp_results, int *cuda_results, int data_size, int num_threads, int blocks_per_grid, int rank);
void check_procs(int size, int expected_num);
int* calc_hist_openmp(int *data, int data_size, int num_threads);
int* calc_hist_cuda(int *data, int *blocks_per_grid, int data_size);
void calc_final_result(int num_threads, int blocks_per_grid, int result[], int *omp_results, int *cuda_results);

int main(int argc, char *argv[]) {
    int *data;
    int data_size = 300000; 
    int part;
    int size, rank;
    int result[HISTOGRAM_SIZE] = { 0 };

    // Initialize MPI
    initialize_mpi(&argc, &argv, &size, &rank);

    // Master process generates random array and sends part of it to worker
    if (rank == MASTER) {
        data = generate_random_array(data_size);
        divide_and_send(data, data_size, data_size / 2, rank);
    } else {
        // Worker process receives data from master
        receive_data(&data, &part, rank);
    }

    // Each process performs histogram calculations
    int num_threads = 4;
    int *omp_results = calc_hist_openmp(data, part, num_threads);
    int blocks_per_grid;
    int *cuda_results = calc_hist_cuda(data, &blocks_per_grid, part);

    // Finalize MPI and print results
    finalize_mpi(data, result, omp_results, cuda_results, data_size, num_threads, blocks_per_grid, rank);

    MPI_Finalize();
    return 0;
}

// Initializes MPI environment
void initialize_mpi(int *argc, char ***argv, int *size, int *rank) {
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, size);
    check_procs(*size, 2); // Check if running with two processes
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    srand(time(NULL)); // Seed random number generator
}

// Generates a random array of specified size
int* generate_random_array(int data_size) {
    int *data = (int*) malloc(data_size * sizeof(int));
    for (int i = 0; i < data_size; ++i) {
        data[i] = rand() % 256; // Generate a random number between 0 and 255
    }
    return data;
}

// Divides data and sends part to worker process
void divide_and_send(int *data, int data_size, int part, int rank) {
    MPI_Send(&part, 1, MPI_INT, WORKER, 0, MPI_COMM_WORLD);
    MPI_Send(data + part, part, MPI_INT, WORKER, 0, MPI_COMM_WORLD);
}

// Receives data from master process
void receive_data(int **data, int *part, int rank) {
    MPI_Status status;
    MPI_Recv(part, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
    *data = (int*) malloc(*part * sizeof(int));
    MPI_Recv(*data, *part, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
}

// Finalizes MPI and prints results
void finalize_mpi(int *data, int *result, int *omp_results, int *cuda_results, int data_size, int num_threads, int blocks_per_grid, int rank) {
    if (rank == MASTER) {
        MPI_Status status;
        MPI_Recv(result, HISTOGRAM_SIZE, MPI_INT, WORKER, 0, MPI_COMM_WORLD, &status);
        calc_final_result(num_threads, blocks_per_grid, result, omp_results, cuda_results);
        int result_sum = 0;
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            result_sum += result[i];
        }
        if (result_sum != data_size)
            printf("Wrong solution %d\n", result_sum);
        else
            printf("Test passed\n");
        free(data);
        free(omp_results);
        free(cuda_results);
    } else {
        calc_final_result(num_threads, blocks_per_grid, result, omp_results, cuda_results);
        MPI_Send(result, HISTOGRAM_SIZE, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
    }
}

// Checks if the number of processes matches the expected number
void check_procs(int size, int expected_num) {
    if (size != expected_num) {
        printf("Run with two processes only\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
}

// Calculates histogram using OpenMP
int* calc_hist_openmp(int *data, int data_size, int num_threads) {
    int thread_id;
    int openmp_part = data_size / 2; // Half of data_size with OpenMP
    int *omp_results = (int*) malloc(num_threads * HISTOGRAM_SIZE * sizeof(int));
    memset(omp_results, 0, num_threads * HISTOGRAM_SIZE * sizeof(int));

    #pragma omp parallel private(thread_id) num_threads(num_threads)
    {
        thread_id = omp_get_thread_num();
        int start = thread_id * (openmp_part / num_threads); // Each thread start after the part of the last one
        int end = (thread_id + 1) * (openmp_part / num_threads);
        for (int i = start; i < end; i++) {
            for (int j = 0; j < HISTOGRAM_SIZE; j++) {
                if (data[i] == j) {
                    omp_results[thread_id * HISTOGRAM_SIZE + j] += 1;
                }
            }
        }
    }

    return omp_results;
}

// Calculates histogram using CUDA
int* calc_hist_cuda(int *data, int *blocks_per_grid, int data_size) {
    int cuda_part = data_size / 2; // Half of data_size with CUDA
    int threads_per_block = 20;
    *blocks_per_grid = 10;
    int *cuda_results = (int*) malloc(*blocks_per_grid * HISTOGRAM_SIZE * sizeof(int));

    if (hist_with_cuda(data + data_size/2, cuda_results, cuda_part,
            threads_per_block, *blocks_per_grid) != 0)
        MPI_Abort(MPI_COMM_WORLD, __LINE__);

    return cuda_results;
}

// Combines histogram results from OpenMP and CUDA
void calc_final_result(int num_threads, int blocks_per_grid, int result[], int *omp_results, int *cuda_results) {
    for (int i = 0; i < num_threads; i++) {
        for (int j = 0; j < HISTOGRAM_SIZE; j++) {
            result[j] += omp_results[i * HISTOGRAM_SIZE + j];
        }
    }
    for (int i = 0; i < blocks_per_grid; i++) {
        for (int j = 0; j < HISTOGRAM_SIZE; j++) {
            result[j] += cuda_results[i * HISTOGRAM_SIZE + j];
        }
    }
}

