#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <time.h>

#define THREADS_PER_BLOCK 100
#define NUM_BLOCKS 1

using namespace std;
typedef double HighlyPrecise;

/**
 * TODO: Don't know why this does not give good results on gene length >= 15.
 * Most probably random numbers error.
 */
const int GENOME_LENGTH = 14;

const float MUTATION_FACTOR = 0.2;
const float CROSSOVER_RATE = 0.6;

const int NUM_EPOCHS = 5000;

struct Chromosome {
	HighlyPrecise genes[GENOME_LENGTH];
	HighlyPrecise fitnessValue;
};

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

__device__ void bubbleSort(Chromosome a[]) {
	int n = THREADS_PER_BLOCK;
	Chromosome temp;
	for (int i = 0; i < n; i++) {
		bool changed = false;
		for (int j = 0; j < n - 1 - i; j++) {
			if (a[j].fitnessValue > a[j + 1].fitnessValue) {
				temp = a[j + 1];
				a[j + 1] = a[j];
				a[j] = temp;
				changed = true;
			}
		}
		if (!changed) {
			// ADD OPTIMIZATION.
			// break;
		}
	}
}

__device__ void printBlockPopulation(Chromosome blockPopulation[]) {
	for (int i = 0; i < THREADS_PER_BLOCK; i++) {
		printf("Fitness: %lf | Chromosome: ", blockPopulation[i].fitnessValue);
		for (int j = 0; j < GENOME_LENGTH; j++) {
			printf("%.02lf ,", blockPopulation[i].genes[j]);
		}
		printf("\n");
	}
}

__global__ void geneticAlgorithm(curandState* states) {
	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	curandState randomState = states[threadIndex];

	__shared__ Chromosome blockPopulation[THREADS_PER_BLOCK];

	HighlyPrecise chromosome[GENOME_LENGTH];

	for (int i = 0; i < GENOME_LENGTH; i++) {
		chromosome[i] = 2.0 * curand_uniform(&randomState) - 1;
		blockPopulation[threadIdx.x].genes[i] = chromosome[i];
	}
	blockPopulation[threadIdx.x].fitnessValue = getFitnessValue(chromosome);

	__syncthreads();

	for (int z = 0; z < NUM_EPOCHS; z++) {

		if ((threadIdx.x == 0) == 1) {
//			printf("==> Before sorting:\n");
//			printBlockPopulation(blockPopulation);
			bubbleSort(blockPopulation);
//			printf("  ==> After sorting:\n");
//			printBlockPopulation(blockPopulation);
		}

		__syncthreads();

		int num_parents = THREADS_PER_BLOCK * (1 - CROSSOVER_RATE);
		if (threadIdx.x >= num_parents) {
//			printf("threadIdx = %d\n", threadIdx.x);
			int maleIndex = curand_uniform(&randomState) * num_parents;
			int femaleIndex = curand_uniform(&randomState) * num_parents;

//			printf("Inside, \tmale:%d = %d \tfemale:%d = %d\n", maleIndex, maleIndex%num_parents, femaleIndex, femaleIndex % num_parents);

			if (maleIndex == femaleIndex) {
				continue;
			}

			Chromosome male = blockPopulation[maleIndex];
			Chromosome female = blockPopulation[femaleIndex];
			Chromosome offspring;
			for (int i = 0; i < GENOME_LENGTH; i++) {
				offspring.genes[i] =
						(i < GENOME_LENGTH / 2) ?
								male.genes[i] : female.genes[i];
			}

			HighlyPrecise random0To1 = curand_uniform(&randomState);
			if (MUTATION_FACTOR > random0To1) {
				for (int i = 0; i < GENOME_LENGTH; i++) {
					HighlyPrecise multiplier = (2.0
							* curand_uniform(&randomState) - 1) / 10;
					if (multiplier < -0.1 || multiplier > 0.1) {
						printf("Invalid multiplier: %lf", multiplier);
					}
					offspring.genes[i] *= multiplier;
				}
			}
			offspring.fitnessValue = getFitnessValue(offspring.genes);
//			if (offspring.fitnessValue == male.fitnessValue
//					|| offspring.fitnessValue == female.fitnessValue) {
//				printf("Baccha on maa baap %lf\n", offspring.fitnessValue);
//			} else {
//				printf("NOT on\n");
//			}
			blockPopulation[threadIdx.x] = offspring;
		}
		__syncthreads();
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		printBlockPopulation(blockPopulation);
		printf("Epochs have been completed. Here's the block's best output:");
		bubbleSort(blockPopulation);
		for (int j = 0; j < GENOME_LENGTH; j++) {
			printf("%lf ", blockPopulation[0].genes[j]);
		}
		printf("\nFitness:%e\n", blockPopulation[0].fitnessValue);
	}
	__syncthreads();
}

int main() {
	int NUM_TOTAL_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;

	curandState *d_randomStates = NULL;
	cudaMalloc((void**) &d_randomStates,
			NUM_TOTAL_THREADS * sizeof(curandState));
	setupRandomStream<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(time(NULL),
			d_randomStates);
	cudaDeviceSynchronize();
	printf("%s", cudaGetErrorString(cudaGetLastError()));

	geneticAlgorithm<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_randomStates);

// Freeing the resources.
	cudaDeviceSynchronize();
	printf("%s", cudaGetErrorString(cudaGetLastError()));
	cudaFree(d_randomStates);

	return 0;
}
