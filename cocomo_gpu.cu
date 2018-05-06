#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <time.h>
using namespace std;

#define THREADS_PER_BLOCK 50
#define NUM_BLOCKS 1

typedef double HighlyPrecise;

const int GENOME_LENGTH = 2;
const float GENE_MAX = 5.0f;
const float GENE_MIN = 0.0;

// TODO: Adjust as per CPU code.
const float MUTATION_FACTOR = 0.8;
const float CROSSOVER_RATE = 0.4;

const int NUM_EPOCHS = 50;

const int NUM_PROJECTS = 12;

struct Chromosome {
	HighlyPrecise genes[GENOME_LENGTH];
	HighlyPrecise fitnessValue;
};

struct Project {
	HighlyPrecise kloc;
	HighlyPrecise actualEffort;
};

Project *h_projects = NULL;

void checkForCudaErrors() {
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		printf("Warning: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
}

Project* loadDatasetIntoGPU() {
	Project *d_projects = NULL;
	// Allocate memory.
	h_projects = (Project*) malloc(NUM_PROJECTS * sizeof(Project));
	cudaMalloc((void**) &d_projects, NUM_PROJECTS * sizeof(Project));
	cudaDeviceSynchronize();
	checkForCudaErrors();

	FILE *fp = fopen("cocomo.dataset", "r");
	if (fp == NULL) {
		printf("ERROR: Unable to open dataset file.");
		return NULL;
	}
//	printf("==> Training data:\nKLOC\t\tActualEffort\n");
	for (int i = 0; i < NUM_PROJECTS && !feof(fp); i++) {
		fscanf(fp, "%lf %lf", &h_projects[i].kloc, &h_projects[i].actualEffort);
//		printf("%lf\t%lf\n", h_projects[i].kloc, h_projects[i].actualEffort);
	}
	fclose(fp);
	cudaMemcpy(d_projects, h_projects, sizeof(Project) * NUM_PROJECTS, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	checkForCudaErrors();
	printf("Input dataset has been copied to GPU successfully.\n");
	return d_projects;
}

void printChromosome(Chromosome a, bool withProjects = false) {
	printf("--> Chromosome: (A:%lf, B:%lf)  ", a.genes[0], a.genes[1]);
	float avgDriftPercentage = 0.0;
	if (withProjects) {
		printf("\n");
	}
	for (int i = 0; i < NUM_PROJECTS; i++) {
		HighlyPrecise estimatedEffort = a.genes[0] * pow(h_projects[i].kloc, a.genes[1]);
		HighlyPrecise error = fabs(h_projects[i].actualEffort - estimatedEffort);
		float percentDiff = 100 * error / h_projects[i].actualEffort;
		avgDriftPercentage += percentDiff;
		if (withProjects) {
			printf("Kloc:%lf, Actual:%lf, Estimated:%lf, Error:%lf, PercentDrift:%f\n",
					h_projects[i].kloc, h_projects[i].actualEffort, estimatedEffort, error,
					percentDiff);
		}
	}
	avgDriftPercentage /= NUM_PROJECTS;
	printf("AverageDriftPercentage: %f\n", avgDriftPercentage);
}

__global__ void setupRandomStream(unsigned int seed, curandState* states) {
	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, threadIndex, 0, &states[threadIndex]);
}

__device__ void printChromosomeOnDevice(Chromosome a, Project projects[],
		bool withProjects = false) {
	printf("--> Device Chromosome: (A:%lf, B:%lf)  ", a.genes[0], a.genes[1]);
	float avgDriftPercentage = 0.0;
	if (withProjects) {
		printf("\n");
	}
	for (int i = 0; i < NUM_PROJECTS; i++) {
		HighlyPrecise estimatedEffort = a.genes[0] * pow(projects[i].kloc, a.genes[1]);
		HighlyPrecise error = fabs(projects[i].actualEffort - estimatedEffort);
		float percentDiff = 100 * error / projects[i].actualEffort;
		avgDriftPercentage += percentDiff;
		if (withProjects) {
			printf("Kloc:%lf, Actual:%lf, Estimated:%lf, Error:%lf, PercentDrift:%f\n",
					projects[i].kloc, projects[i].actualEffort, estimatedEffort, error,
					percentDiff);
		}
	}
	avgDriftPercentage /= NUM_PROJECTS;
	printf("AverageDriftPercentage: %f\n", avgDriftPercentage);
}


/**
 * Prints the whole population of a block from the shared memory.
 */
__device__ void printBlockPopulation(Chromosome blockPopulation[], Project projects[]) {
	for (int i = 0; i < blockDim.x; i++) {
		printf("--ID:%d ", i);
		printChromosomeOnDevice(blockPopulation[i], projects, false);
	}
}

__device__ HighlyPrecise getEstimatedEffort(HighlyPrecise genes[], Project project) {
	return genes[0] * pow(project.kloc, genes[1]);
}

__device__ HighlyPrecise getFitnessValue(HighlyPrecise genes[], Project projects[]) {
	HighlyPrecise fitnessValue = 0;
	for (int i = 0; i < NUM_PROJECTS; i++) {
		HighlyPrecise differenceInEfforts = fabs(
				projects[i].actualEffort - getEstimatedEffort(genes, projects[i]));
		fitnessValue += differenceInEfforts;// * differenceInEfforts;
	}
	printf("D_Fitness: %lf\n", fitnessValue);
	return fitnessValue;
}

__device__ void initializeBlockPopulation(Chromosome blockPopulation[], curandState* randomState,
		Project projects[]) {
	HighlyPrecise chromosome[GENOME_LENGTH];
	for (int i = 0; i < GENOME_LENGTH; i++) {
		// Changed because 0.0 to 10.0
//		chromosome[i] = curand(randomState) % GENE_MAX;
		chromosome[i] = curand_uniform_double(randomState) * GENE_MAX;
		blockPopulation[threadIdx.x].genes[i] = chromosome[i];
	}
	blockPopulation[threadIdx.x].fitnessValue = getFitnessValue(chromosome, projects);
	printf("Initialized ");
	printChromosomeOnDevice(blockPopulation[threadIdx.x], projects, false);
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

__device__ Chromosome crossover(Chromosome blockPopulation[], curandState* randomState,
		int num_parents) {
	// Choosing parents.
	int maleIndex = curand_uniform(randomState) * num_parents;
	int femaleIndex = curand_uniform(randomState) * num_parents;
	if (maleIndex == femaleIndex) {
		return blockPopulation[threadIdx.x];
	}
	Chromosome male = blockPopulation[maleIndex];
	Chromosome female = blockPopulation[femaleIndex];
	Chromosome offspring;

	offspring.genes[0] = male.genes[0];
	offspring.genes[1] = female.genes[1];

	return offspring;
}

__device__ void mutate(Chromosome *offspring, curandState* randomState) {
	for (int i = 0; i < GENOME_LENGTH; i++) {
		// XXX: Changed. Multiplier in this is from 0 to 1.
		HighlyPrecise multiplier = curand_uniform_double(randomState);
		offspring->genes[i] *= multiplier;
	}
}

__device__ void startIteration(Chromosome blockPopulation[], curandState* randomState,
		Project projects[]) {

	int num_parents = ceil(blockDim.x * (1 - CROSSOVER_RATE));

	// Start choosing parents and fill the remaining.
	if (threadIdx.x >= num_parents) {
		// Crossover.
		Chromosome offspring = crossover(blockPopulation, randomState, num_parents);
		// Mutation.
		if (MUTATION_FACTOR > curand_uniform_double(randomState)) {
			mutate(&offspring, randomState);
		}
		// Evaluation.
		offspring.fitnessValue = getFitnessValue(offspring.genes, projects);
		// Updation in population.
		blockPopulation[threadIdx.x] = offspring;
	}
}

/**
 * Core genetic algorithm.
 */
__global__ void geneticAlgorithm(bool freshRun, Chromosome *d_inputPopulation, curandState* states,
		Project *projects, Chromosome *d_outputPopulation) {

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
		initializeBlockPopulation(blockPopulation, &randomState, projects);
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

		startIteration(blockPopulation, &randomState, projects);
		__syncthreads();
		printf("\n=>GPU Iteration %d has completed:", z);
		printBlockPopulation(blockPopulation, projects);
		printf("=Iteration %d has printed\n", z);
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

/**
 * Main Function.
 */
int main() {
	int NUM_TOTAL_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;

	// Setup the random number generation stream on device.
	curandState *d_randomStates = NULL;
	cudaMalloc((void**) &d_randomStates, NUM_TOTAL_THREADS * sizeof(curandState));
	setupRandomStream<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(time(NULL), d_randomStates);
	cudaDeviceSynchronize();
	checkForCudaErrors();
	// Load dataset into GPU.
	Project *d_projects = loadDatasetIntoGPU();
	// Allocate memory for output on GPU.
	Chromosome *h_gpuOut = NULL;
	Chromosome *d_outputPopulation = NULL;
	h_gpuOut = (Chromosome*) malloc(NUM_BLOCKS * sizeof(Chromosome));
	cudaMalloc((void**) &d_outputPopulation, NUM_BLOCKS * sizeof(Chromosome));
	cudaDeviceSynchronize();
	checkForCudaErrors();
	// STAGE 1 EXECUTION.
	geneticAlgorithm<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(true, NULL, d_randomStates, d_projects,
			d_outputPopulation);
	// Copy output back to host.
	cudaMemcpy(h_gpuOut, d_outputPopulation, sizeof(Chromosome) * NUM_BLOCKS,
			cudaMemcpyDeviceToHost);

	// Allocate memory for stage 2 input on GPU.
	Chromosome *d_inputPopulation;
	cudaMalloc((void**) &d_inputPopulation, sizeof(Chromosome) * NUM_BLOCKS);
	cudaDeviceSynchronize();
	checkForCudaErrors();

	printf("==> STAGE 1 output:\nWinning chromosomes:\n ");
	for (int i = 0; i < NUM_BLOCKS; i++) {
		printChromosome(h_gpuOut[i], true);
	}
	printf("\n");

//	cudaMemcpy(d_inputPopulation, h_gpuOut, sizeof(Chromosome) * NUM_BLOCKS,
//			cudaMemcpyHostToDevice);

//	geneticAlgorithm<<<1, NUM_BLOCKS>>>(false, d_inputPopulation, d_randomStates, d_projects,
//			d_outputPopulation);

//	cudaMemcpy(h_gpuOut, d_outputPopulation, sizeof(Chromosome) * 1, cudaMemcpyDeviceToHost);
//
//	printf("\n======== GPU Results ========\n");
//	printChromosome(h_gpuOut[0], true);
//	printf("\n");

	cudaDeviceSynchronize();
	checkForCudaErrors();

	// Freeing the resources.
	cudaFree(d_randomStates);
	cudaFree(d_outputPopulation);
	cudaFree(d_projects);
	cudaFree(d_inputPopulation);
	return 0;
}




