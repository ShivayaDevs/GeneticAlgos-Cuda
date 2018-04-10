#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 4
#define NUM_BLOCKS 2

using namespace std;
typedef double HighlyPrecise;

const int GENOME_LENGTH = 16;

const double GENE_MIN = -1;
const double GENE_MAX = +1;

const float MUTATION_FACTOR = 0.2;
const float CROSSOVER_RATE = 0.6;

const int NUMBER_CHROMOSOMES = 100;
const int NUM_EPOCHS = 1000;

__global__ void setupRandomStream(unsigned int seed, curandState* states) {
	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	/* Make sure that this is seed and might want to reduce the number of states to threadIdx.x .*/
	curand_init(seed, threadIndex, 0, &states[threadIndex]);
}

__device__ HighlyPrecise getFitnessValue(HighlyPrecise chromosome[]) {
	HighlyPrecise fitnessValue = 0;
	for (int i = 0; i < GENOME_LENGTH; i++) {
		fitnessValue += chromosome[i] * chromosome[i];
	}
	return fitnessValue;
}

__global__ void geneticAlgorithm(curandState* states) {
	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	curandState randomState = states[threadIndex];

	__shared__ HighlyPrecise blockPopulation[THREADS_PER_BLOCK][GENOME_LENGTH];
	__shared__ HighlyPrecise fitnessValues[THREADS_PER_BLOCK];

	HighlyPrecise chromosome[GENOME_LENGTH];

	for (int i = 0; i < GENOME_LENGTH; i++) {
		chromosome[i] = 2.0 * curand_uniform(&randomState) - 1;
		blockPopulation[threadIdx.x][i] = chromosome[i];
	}
	// By now random population for this block has been generated.

	fitnessValues[threadIdx.x] = getFitnessValue(chromosome);

	printf("%d -> %d Syncing threads...\n", blockIdx.x, threadIdx.x);

	__syncthreads();




	if (threadIdx.x == 0) {
		printf("Since, I'm thread with Id=0, here's the shared memory: ");
		for (int i = 0; i < THREADS_PER_BLOCK; i++) {
			for (int j = 0; j < GENOME_LENGTH; j++) {
				printf("%lf ", blockPopulation[i][j]);
			}
			printf("\nFitness:%e\n", fitnessValues[i]);
		}
	}
}

int main() {
	int NUM_TOTAL_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;

	curandState *d_randomStates = NULL;
	cudaMalloc((void**) &d_randomStates,
			NUM_TOTAL_THREADS * sizeof(curandState));
	setupRandomStream<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(1, d_randomStates);
	cudaDeviceSynchronize();
	cudaGetErrorString(cudaGetLastError());

	geneticAlgorithm<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_randomStates);

	// Freeing the resources.
	cudaDeviceSynchronize();
	cudaGetErrorString(cudaGetLastError());
	cudaFree(d_randomStates);

	return 0;
}
