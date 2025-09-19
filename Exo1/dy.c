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

// Function for the master process
double masterProcess(int size)
{
    double answer = 0;
    int* arr = (int*)calloc(SIZE, sizeof(int));
    for (int i = 0; i < SIZE; i++)
        arr[i] = i;

    int send = 0, recv = 0;
    MPI_Status status;

    // Initial work
    int smaller = size < SIZE ? size : SIZE;
    for (int i = 1; i < smaller; i++)
    {
        MPI_Send(&arr[send], 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        send++;
    }

    do
    {
        double tmp;
        MPI_Recv(&tmp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        recv++;
        if (recv <= SIZE) // There is still work to perform
        {
            answer += tmp;
            printf("answer of %d is %e\n", recv, answer);
            MPI_Send(&send, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
            send++;
        }
        else    // There is no work to do
        {
            printf("No more work to do.\n");
            // Send termination tag to all slaves
            for (int i = 1; i < smaller; i++)
                MPI_Send(&send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } while (recv <= SIZE);

    free(arr);
    return answer;
}

// Function for the worker process
void workerProcess()
{
    MPI_Status status;
    int x;
    double answer;
    while (1)
    {
        MPI_Recv(&x, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (!status.MPI_TAG)
            break;
        answer = 0;
        for (int i = 0; i < SIZE; i++)
            answer += heavy(x, i);
        MPI_Send(&answer, 1, MPI_DOUBLE, 0, status.MPI_TAG, MPI_COMM_WORLD);
        printf("Result of %d with tag %d sent\n", x, status.MPI_TAG);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start, answer;

    if (rank == 0)
    {
        start = MPI_Wtime();
        answer = masterProcess(size);
    }
    else
    {
        workerProcess();
    }

    if (rank == 0)
    {
        printf("Final answer = %e\n", answer);
        double end = MPI_Wtime();
        printf("Execution time: %.3fs\n", (end - start));
    }

    MPI_Finalize();
    return 0;
}
