build:
	mpicxx -fopenmp -c main.c -o main.o
	nvcc -I./Common  -gencode arch=compute_61,code=sm_61 -c cudaFuncs.cu -o cudaFuncs.o
	mpicxx -fopenmp -o calcHistoParallelHybrid  main.o  cudaFuncs.o  -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -f *.o ./calcHistoParallelHybrid

run:
	mpiexec -np 2 ./calcHistoParallelHybrid

