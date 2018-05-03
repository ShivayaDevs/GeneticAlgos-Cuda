#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <time.h>
using namespace std;

#define THREADS_PER_BLOCK 32
#define NUM_BLOCKS 32

typedef double HighlyPrecise;

const int GENOME_LENGTH = 14;
const int GENE_MAX = 2048;

const float MUTATION_FACTOR = 0.2;
const float CROSSOVER_RATE = 0.6;

const int NUM_EPOCHS = 1000;

struct Chromosome {
	HighlyPrecise genes[GENOME_LENGTH];
	HighlyPrecise fitnessValue;
};

__global__ void setupRandomStream(unsigned int seed, curandState* states) {
	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, threadIndex, 0, &states[threadIndex]);
}

__device__ HighlyPrecise getFitnessValue(HighlyPrecise chromosome[]) {
	HighlyPrecise value = 0.0;
	HighlyPrecise cosProducts = 1.0;
	for (int i = 0; i < GENOME_LENGTH; i++) {
		value += chromosome[i] * chromosome[i];
		cosProducts *= cos(chromosome[i] / sqrtf(i+1));
	}
	return value / 4000 - cosProducts + 1;
}

/**
 * Sorts the population that is present in the shared memory of one block.
 * Please note that this sorting is sequential.
 */
__device__ void bubbleSort(Chromosome a[]) {
	int n = blockDim.x;
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
			break;
		}
	}
}

__device__ void printChromosome(Chromosome c) {
	printf("Fitness: %lf | Chromosome: ", c.fitnessValue);
	for (int j = 0; j < GENOME_LENGTH; j++) {
		printf("%lf ,", c.genes[j]);
	}
	printf("\n");
}

/**
 * Prints the whole population of a block from the shared memory.
 */
__device__ void printBlockPopulation(Chromosome blockPopulation[]) {
	for (int i = 0; i < blockDim.x; i++) {
		printChromosome(blockPopulation[i]);
	}
}

__device__ void initializeBlockPopulation(Chromosome blockPopulation[],
		curandState* randomState) {
	HighlyPrecise chromosome[GENOME_LENGTH];
	for (int i = 0; i < GENOME_LENGTH; i++) {
		chromosome[i] = GENE_MAX * (2.0 * curand_uniform(randomState) - 1);
		blockPopulation[threadIdx.x].genes[i] = chromosome[i];
	}
	blockPopulation[threadIdx.x].fitnessValue = getFitnessValue(chromosome);
}

__device__ Chromosome crossover(Chromosome blockPopulation[],
		curandState* randomState, int num_parents) {
	// Choosing parents.
	int maleIndex = curand_uniform(randomState) * num_parents;
	int femaleIndex = curand_uniform(randomState) * num_parents;
	if (maleIndex == femaleIndex) {
		return blockPopulation[threadIdx.x];
	}
	Chromosome male = blockPopulation[maleIndex];
	Chromosome female = blockPopulation[femaleIndex];
	Chromosome offspring;

	for (int i = 0; i < GENOME_LENGTH; i++) {
		offspring.genes[i] =
				(i < GENOME_LENGTH / 2) ? male.genes[i] : female.genes[i];
	}
	return offspring;
}

__device__ void mutate(Chromosome *offspring, curandState* randomState) {
	for (int i = 0; i < GENOME_LENGTH; i++) {
		HighlyPrecise multiplier = (2.0 * curand_uniform(randomState) - 1) / 10;
		offspring->genes[i] *= multiplier;
	}
}

__device__ void startIteration(Chromosome blockPopulation[],
		curandState* randomState) {

	int num_parents = blockDim.x * (1 - CROSSOVER_RATE);

	// Start choosing parents and fill the remaining.
	if (threadIdx.x >= num_parents) {
		// Crossover.
		Chromosome offspring = crossover(blockPopulation, randomState,
				num_parents);
		// Mutation.
		if (MUTATION_FACTOR > curand_uniform(randomState)) {
			mutate(&offspring, randomState);
		}
		// Evaluation.
		offspring.fitnessValue = getFitnessValue(offspring.genes);
		// Updation in population.
		blockPopulation[threadIdx.x] = offspring;
	}
}
//
//__device__ setupBlockPopulation(Chromosome blockPopulation[], curandState* states) {
//
//}

/**
 * Core genetic algorithm.
 */
__global__ void geneticAlgorithm(bool freshRun, Chromosome *d_inputPopulation,
		curandState* states, Chromosome *d_outputPopulation) {

	/* This won't be the same size when running in stage 2,
	 * i.e. Block's bests conpetiting against each other.
	 * But since CUDA doesn't allow (in an easy way) to dynamically decide the size,
	 * it has been kept to the larger one. BUT, blocks can be more than threads.*/
	__shared__ Chromosome blockPopulation[THREADS_PER_BLOCK];

	int blockIndex = blockIdx.x;
	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	curandState randomState = states[threadIndex];

	if (freshRun) {
		// Because this is a stage 1 run, we need to initialize the random population on GPU.
		initializeBlockPopulation(blockPopulation, states);
	} else {
		blockPopulation[threadIdx.x] = d_inputPopulation[threadIndex];
	}

	// Barrier ensures that population is available on the block in whatever way.
	__syncthreads();

	for (int z = 0; z < NUM_EPOCHS; z++) {

		if (threadIdx.x == 0) {
			bubbleSort(blockPopulation);
		}
		__syncthreads();
		// Chromosomes orted in the increasing order of fitness function.

		startIteration(blockPopulation, &randomState);
		__syncthreads();
	}

	// all threads of this block have completed.
	__syncthreads();

	if (threadIdx.x == 0) {
//		printBlockPopulation(blockPopulation);
//		printf("Epochs have been completed. Here's the block's best output:\n");
		bubbleSort(blockPopulation);
//		printChromosome(blockPopulation[0]);

// Copy these results to the global memory.
		d_outputPopulation[blockIndex] = blockPopulation[0];
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
	printf("CudaStatus: %s\n", cudaGetErrorString(cudaGetLastError()));

	Chromosome *h_gpuOut = NULL;
	Chromosome *d_outputPopulation = NULL;
	h_gpuOut = (Chromosome*) malloc(NUM_BLOCKS * sizeof(Chromosome));
	cudaMalloc((void**) &d_outputPopulation, NUM_BLOCKS * sizeof(Chromosome));
	cudaDeviceSynchronize();
	printf("CudaStatus: Allocating d_out: %s\n",
			cudaGetErrorString(cudaGetLastError()));

	geneticAlgorithm<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(true, NULL,
			d_randomStates, d_outputPopulation);

	cudaMemcpy(h_gpuOut, d_outputPopulation, sizeof(Chromosome) * NUM_BLOCKS,
			cudaMemcpyDeviceToHost);

//	stage2
	Chromosome *d_inputPopulation;
	cudaMalloc((void**) &d_inputPopulation, sizeof(Chromosome) * NUM_BLOCKS);
	cudaDeviceSynchronize();
	printf("CudaStatus: Allocating d_input: %s\n",
			cudaGetErrorString(cudaGetLastError()));

	for (int i = 0; i < NUM_BLOCKS; i++) {
//		printf("block %d output:%lf\n", i, h_gpuOut[i].fitnessValue);
	}
	cudaMemcpy(d_inputPopulation, h_gpuOut, sizeof(Chromosome) * NUM_BLOCKS,
			cudaMemcpyHostToDevice);

	geneticAlgorithm<<<1, NUM_BLOCKS>>>(false, d_inputPopulation,
			d_randomStates, d_outputPopulation);

	cudaMemcpy(h_gpuOut, d_outputPopulation, sizeof(Chromosome) * 1,
			cudaMemcpyDeviceToHost);
	printf("\n\n===========================\n");
	printf("|| Final result:%e\n", h_gpuOut[0].fitnessValue);
	for (int i = 0; i < GENOME_LENGTH; i++) {
		printf("%e ", h_gpuOut[0].genes[i]);
	}
	printf("\n");
// Freeing the resources.
	cudaDeviceSynchronize();
	printf("CudaStatus: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaFree(d_randomStates);
	cudaFree(d_outputPopulation);
	return 0;
}
