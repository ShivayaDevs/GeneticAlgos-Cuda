#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <time.h>
using namespace std;

#define THREADS_PER_BLOCK 10
#define NUM_BLOCKS 2

typedef double HighlyPrecise;

const int GENOME_LENGTH = 2;
const int GENE_MAX = 10.0;
const int GENE_MIN = 0.0;

// TODO: Adjust as per CPU code.
const float MUTATION_FACTOR = 0.8;
const float CROSSOVER_RATE = 0.4;

const int NUM_EPOCHS = 1500;

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
	cudaMemcpy(d_projects, h_projects, sizeof(Project) * NUM_PROJECTS,
			cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	checkForCudaErrors();
	printf("Input dataset has been copied to GPU successfully.\n");
	return d_projects;
}


/**
 * Main Function.
 */
int main() {
	int NUM_TOTAL_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;

	// Setup the random number generation stream on device.
	curandState *d_randomStates = NULL;
	cudaMalloc((void**) &d_randomStates,
			NUM_TOTAL_THREADS * sizeof(curandState));
	setupRandomStream<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(time(NULL),
			d_randomStates);
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
	geneticAlgorithm<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(true, NULL,
			d_randomStates, d_projects, d_outputPopulation);

	cudaMemcpy(h_gpuOut, d_outputPopulation, sizeof(Chromosome) * NUM_BLOCKS,
			cudaMemcpyDeviceToHost);

	//	stage2
	Chromosome *d_inputPopulation;
	cudaMalloc((void**) &d_inputPopulation, sizeof(Chromosome) * NUM_BLOCKS);
	cudaDeviceSynchronize();
	checkForCudaErrors();

	for (int i = 0; i < NUM_BLOCKS; i++) {
//		printf("block %d output:%lf\n", i, h_gpuOut[i].fitnessValue);
	}
	cudaMemcpy(d_inputPopulation, h_gpuOut, sizeof(Chromosome) * NUM_BLOCKS,
			cudaMemcpyHostToDevice);

	geneticAlgorithm<<<1, NUM_BLOCKS>>>(false, d_inputPopulation,
			d_randomStates, d_dataset, d_outputPopulation);

	cudaMemcpy(h_gpuOut, d_outputPopulation, sizeof(Chromosome) * 1,
			cudaMemcpyDeviceToHost);

	printf("\n======== GPU Results ========\n");
	print(h_gpuOut[0].genes, h_projects);
	printf("\n");

	cudaDeviceSynchronize();
	checkForCudaErrors();

	// Freeing the resources.
	cudaFree(d_randomStates);
	cudaFree(d_outputPopulation);
	return 0;
}



























__global__ void setupRandomStream(unsigned int seed, curandState* states) {
	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, threadIndex, 0, &states[threadIndex]);
}

__device__ HighlyPrecise getEstimatedEffort(HighlyPrecise genes[],
		int projectId, Project projects[]) {
	return genes[0] * pow(projects[projectId].kloc, genes[1]);
}

__device__ HighlyPrecise getFitnessValue(HighlyPrecise genes[],
		Project projects[]) {

	for (int i = 0; i < NUM_PROJECTS; i++) {
		printf(
				"Project %d : Kloc=%lf Actual=%lf Estimated=%lf genes=(%lf, %lf)\n",
				i, projects[i].kloc, projects[i].actualEffort,
				getEstimatedEffort(genes, i, projects), genes[0], genes[1]);
	}

	HighlyPrecise fitnessValue = 0;
	for (int i = 0; i < NUM_PROJECTS; i++) {
		HighlyPrecise differenceInEfforts = fabs(
				projects[i].actualEffort
						- getEstimatedEffort(genes, i, projects));
		fitnessValue += differenceInEfforts * differenceInEfforts;
	}
	return fitnessValue;
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
		curandState* randomState, Project projects[]) {
	HighlyPrecise chromosome[GENOME_LENGTH];
	for (int i = 0; i < GENOME_LENGTH; i++) {
		// Changed because 0.0 to 10.0
		chromosome[i] = curand_uniform(randomState) * GENE_MAX;
		blockPopulation[threadIdx.x].genes[i] = chromosome[i];
	}
	blockPopulation[threadIdx.x].fitnessValue = getFitnessValue(chromosome,
			projects);
	printf("PI: Fitness=%lf, Chromosome=(%lf %lf)\n",
			blockPopulation[threadIdx.x].fitnessValue, chromosome[0],
			chromosome[1]);
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
		// XXX: Changed. Multiplier in this is from 0 to 1.
		HighlyPrecise multiplier = curand_uniform(randomState);
		offspring->genes[i] *= multiplier;
	}
}

__device__ void startIteration(Chromosome blockPopulation[],
		curandState* randomState, Project projects[]) {

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
		offspring.fitnessValue = getFitnessValue(offspring.genes, projects);
		// Updation in population.
		blockPopulation[threadIdx.x] = offspring;
	}
}

/**
 * Core genetic algorithm.
 */
__global__ void geneticAlgorithm(bool freshRun, Chromosome *d_inputPopulation,
		curandState* states, Project *projects,
		Chromosome *d_outputPopulation) {

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
		initializeBlockPopulation(blockPopulation, states, projects);
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

void checkForCudaErrors() {
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		printf("Warning: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
}

void print(HighlyPrecise genes[], Project projects[]) {
	printf("ProjectId\tKloc\tActualEffort\tEstimatedEffortGA\n");
	for (int i = 0; i < NUM_PROJECTS; i++) {
		HighlyPrecise estimatedEffort = genes[0]
				* pow(projects[i].kloc, genes[1]);
		printf("%d\t%lf\t%lf\t%lf\n", i, projects[i].kloc,
				projects[i].actualEffort, estimatedEffort);
	}
	printf("==> Parameters: ");
	cout << "A: " << genes[0] << "\n";
	cout << "B: " << genes[1] << "\n";
}

