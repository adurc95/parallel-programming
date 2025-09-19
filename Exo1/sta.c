#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mpi.h"

#define HEAVY 1000
#define SIZE 60

// Function to perform heavy computations
double heavy(int x, int y) 
{
    int i, loop;
    double sum = 0;
    if (x > 0.25*SIZE && x < 0.5*SIZE && y > 0.4 * SIZE && y < 0.6 * SIZE)
        loop = x * y;
    else
        loop = y + x;

    for (i = 0; i < loop * HEAVY; i++)
        sum += cos(exp(sin((double)i / HEAVY)));

    return sum;
}

// Function to initialize MPI
void initializeMPI(int *argc, char ***argv, int *rank, int *size)
{
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
}

// Function to scatter data
void scatterData(int rank, int size, int *arr, int s, int *myArr)
{
    MPI_Scatter(arr, s, MPI_INT, myArr, s, MPI_INT, 0, MPI_COMM_WORLD);
}

// Function to perform computation
double performComputation(int *myArr, int s)
{
    double answer = 0;
    for (int i = 0; i < s; i++)
    {
        for (int j = 0; j < SIZE; j++)
            answer += heavy(myArr[i], j);
    }
    return answer;
}

// Function to gather results
double gatherResults(int rank, int size, double answer)
{
    MPI_Status status;
    if (rank == 0)
    {
        double tmp;
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            answer += tmp;
        }
    }
    else
        MPI_Send(&answer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    return answer;
}

// Function to finalize MPI
void finalizeMPI()
{
    MPI_Finalize();
}

int main(int argc, char *argv[])
{
    double start;
    int rank, size;
    double answer = 0;
    int* arr = NULL;

    initializeMPI(&argc, &argv, &rank, &size);

    if (rank == 0)
    {
        start = MPI_Wtime();
        arr = (int*)calloc(SIZE, sizeof(int));
        for (int i = 0; i < SIZE; i++)
            arr[i] = i;
    }   

    int s = SIZE/size;
    int myArr[s];
    scatterData(rank, size, arr, s, myArr);

    double partialAnswer = performComputation(myArr, s);
    answer = gatherResults(rank, size, partialAnswer);

    if (rank == 0)
    {
        printf("answer = %e\n", answer);
        double end = MPI_Wtime();
        printf("Execution time: %.3fs\n", (end - start));
    }

    finalizeMPI();
    return 0;
}
