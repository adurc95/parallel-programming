
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILE_NAME "data.txt"
#define ROOT 0

int main(int, char* []);
void initializeMPI(int*, char***, int*, int*);
void broadcastVariables(int*, int*, int*, int*, MPI_Comm);
void processStringsAndSearch(int rank, char** strings, int numProcs, int stringLength, int maxIterations, char* substring, MPI_Comm comm);
char** readStringsFromFile(const char*, int*, int*, int*, char**, int*);
char* getcurrString(int, char**, int, int, MPI_Comm);
int move(int, int, int, int, int, int, int, char*, const char*, int*, MPI_Comm);
char** collectStrings(int, int, int, char*, MPI_Comm);
char** searchSubstring(int, int, int, int, char*, const char*, MPI_Comm);


int main(int argc, char* argv[]) {
    MPI_Comm comm;
    int rank, size, numProcs, stringLength, maxIterations, substringLength;
    char* substring, * procString, ** strings = NULL, ** scrambledStrings = NULL;

    // Initialize MPI
    initializeMPI(&argc, &argv, &rank, &size);

    // Read strings from file if rank is ROOT
    if (rank == ROOT) {
        strings = readStringsFromFile(FILE_NAME, &numProcs, &stringLength, &maxIterations, &substring, &substringLength);
        if (size != numProcs * numProcs) {
            printf("Parsing the file requires %d processes, but the program runs with %d processes\n", numProcs * numProcs, size);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }
    }

    // Broadcast variables to all processes
    broadcastVariables(&numProcs, &stringLength, &maxIterations, &substringLength, MPI_COMM_WORLD);

    // Broadcast substring to all processes
    if (rank != ROOT)
        substring = (char*)malloc(sizeof(char) * (substringLength + 1));
    MPI_Bcast(substring, substringLength + 1, MPI_CHAR, ROOT, MPI_COMM_WORLD);

    // Create cartesian communicator
    int dimensions[2] = { numProcs, numProcs }, periods[2] = { 1, 1 };
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0, &comm);

    // Process strings and search for substring
    processStringsAndSearch(rank, strings, numProcs, stringLength, maxIterations, substring, comm);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

// Initialize MPI
void initializeMPI(int* argc, char*** argv, int* rank, int* size) {
    int provided;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        printf("Error: The required thread support level is not met\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
}

// Broadcast variables to all processes
void broadcastVariables(int* numProcs, int* stringLength, int* maxIterations, int* substringLength, MPI_Comm comm) {
    MPI_Bcast(numProcs, 1, MPI_INT, ROOT, comm);
    MPI_Bcast(stringLength, 1, MPI_INT, ROOT, comm);
    MPI_Bcast(maxIterations, 1, MPI_INT, ROOT, comm);
    MPI_Bcast(substringLength, 1, MPI_INT, ROOT, comm);
}

// Process strings and search for substring
void processStringsAndSearch(int rank, char** strings, int numProcs, int stringLength, int maxIterations, char* substring, MPI_Comm comm) {
    char* procString;
    char** scrambledStrings;

    // Get the string for the current process
    procString = getcurrString(rank, strings, numProcs, stringLength, MPI_COMM_WORLD);

    // Search for substring
    scrambledStrings = searchSubstring(rank, numProcs, stringLength, maxIterations, procString, substring, comm);

    // Print results at ROOT process
    if (rank == ROOT) {
        if (scrambledStrings != NULL) {
            for (int i = 0; i < numProcs * numProcs; i++) {
                printf("%s\n", scrambledStrings[i]);
            }
        }
        else {
            printf("The substring was not found\n");
        }
    }
}

// Read strings from file
char** readStringsFromFile(const char* fileName, int* numProcs, int* stringLength, int* maxIterations, char** substring, int* substringLength) {
    FILE* fp;
    char** strings;

    if ((fp = fopen(fileName, "r")) == NULL) {
        printf("Cannot open file %s for reading\n", fileName);
        return NULL;
    }

    fscanf(fp, "%d %d %d", numProcs, stringLength, maxIterations);
    *substring = (char*)malloc(sizeof(char) * (2 * (*stringLength) + 1));
    if (*substring == NULL) {
        printf("Problem to allocate memory\n");
        return NULL;
    }
    fscanf(fp, "%s", *substring);

    strings = (char**)malloc((*numProcs) * (*numProcs) * sizeof(char*));
    if (strings == NULL) {
        printf("Problem to allocate memory\n");
        return NULL;
    }

    for (int i = 0; i < (*numProcs) * (*numProcs); i++) {
        strings[i] = (char*)malloc((2 * (*stringLength) + 1) * sizeof(char));
        if (strings[i] == NULL) {
            printf("Problem to allocate memory\n");
            return NULL;
        }
        fscanf(fp, "%s", strings[i]);
    }
    *substringLength = strlen(*substring);
    fclose(fp);
    return strings;
}

// Get string for the current process
char* getcurrString(int rank, char** strings, int numProcs, int stringLength, MPI_Comm comm) {
    MPI_Status status;
    char* currString = (char*)malloc((2 * stringLength + 1) * sizeof(char));

    if (rank == ROOT) {
        currString = strings[ROOT];
        for (int i = 1; i < numProcs * numProcs; i++)
            MPI_Send(strings[i], 2 * stringLength + 1, MPI_CHAR, i, 0, comm);
    }
    else
        MPI_Recv(currString, 2 * stringLength + 1, MPI_CHAR, ROOT, 0, comm, &status);

    return currString;
}

// Search for substring
char** searchSubstring(int rank, int numProcs, int stringLength, int maxIterations, char* currString, const char* substring, MPI_Comm comm) {
    int  found = 0, leftRank, rightRank, * foundProcesses = NULL, upRank, downRank;

    char** strings = NULL;

    if (rank == ROOT)
        foundProcesses = (int*)malloc(sizeof(int) * numProcs * numProcs);

    MPI_Cart_shift(comm, 0, 1, &downRank, &upRank);
    MPI_Cart_shift(comm, 1, 1, &leftRank, &rightRank);

    for (int i = 0; i < maxIterations && !found; i++)
        found = move(rank, numProcs, stringLength, upRank, downRank, leftRank, rightRank, currString, substring, foundProcesses, comm);

    if (found)
        strings = collectStrings(rank, numProcs, stringLength, currString, comm);

    return strings;
}

// Move the strings
int move(int rank, int numProcs, int stringLength, int upRank, int downRank, int leftRank, int rightRank, char* currString,
    const char* substring, int* foundProcesses, MPI_Comm comm) {
    MPI_Status status;
    int found;
    char* leftSend = (char*)malloc(sizeof(char) * stringLength);
    char* upSend = (char*)malloc(sizeof(char) * stringLength);

    for (int i = 0; i < stringLength; i++) {
        upSend[i] = currString[2 * i];
        leftSend[i] = currString[2 * i + 1];
    }

    MPI_Sendrecv(leftSend, stringLength, MPI_CHAR, leftRank, 0, currString, stringLength, MPI_CHAR, rightRank, 0, comm, &status);
    MPI_Sendrecv(upSend, stringLength, MPI_CHAR, upRank, 0, currString + stringLength, stringLength, MPI_CHAR, downRank, 0, comm, &status);
    free(upSend);
    free(leftSend);

    found = (strstr(currString, substring) != NULL);

    MPI_Gather(&found, 1, MPI_INT, foundProcesses, 1, MPI_INT, ROOT, comm);
    if (rank == ROOT) {
        for (int i = 0; (i < numProcs * numProcs) && (found == 0); i++)
            if (foundProcesses[i] == 1)
                found = 1;
    }
    MPI_Bcast(&found, 1, MPI_INT, ROOT, comm);

    return found;
}

// Collect strings from all processes
char** collectStrings(int rank, int numProcs, int stringLength, char* currString, MPI_Comm comm) {
    char** strings = NULL, * temp;
    MPI_Status status;

    if (rank == ROOT) {
        temp = (char*)malloc(sizeof(char) * (2 * stringLength + 1));
        temp[2 * stringLength] = '\0';

        strings = (char**)malloc(sizeof(char*) * numProcs * numProcs);
        strings[ROOT] = currString;
        for (int i = 1; i < numProcs * numProcs; i++) {
            MPI_Recv(temp, 2 * stringLength, MPI_CHAR, MPI_ANY_SOURCE, 0, comm, &status);
            strings[status.MPI_SOURCE] = strdup(temp);
        }
    }
    else
        MPI_Send(currString, 2 * stringLength, MPI_CHAR, ROOT, 0, comm);

    return strings;
}
