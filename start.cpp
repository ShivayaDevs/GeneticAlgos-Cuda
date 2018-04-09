#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

const int GENOME_LENGTH = 16;
const int NUMBER_CHROMOSOMES = 1000;

const int GENE_MIN = -1;
const int GENE_MAX = +1;

const int MUTATION_FACTOR = 0.2;
const int CROSSOVER_RATE = 0.6;

const int NUM_EPOCHS = 10000;

class Chromosome {
	float genes[GENOME_LENGTH];

public:

	Chromosome() {
		for (int j = 0; j < GENOME_LENGTH; j++) {
			genes[j] = (2.0f * GENE_MAX * rand()) / RAND_MAX - GENE_MAX;
		}
	}

	Chromosome(float genes[]) {
		memcpy(this->genes, genes, GENOME_LENGTH * sizeof(float));
	}

	float* getGenes() {
		return genes;
	}

	float getFitnessValue() {
		int value = 0.0;
		for (int i = 0; i < GENOME_LENGTH; i++) {
			value += genes[i] * genes[i];
		}
		return value;
	}

	Chromosome crossover(Chromosome b) {
		int mid = GENOME_LENGTH / 2;
		float offspringGenes[GENOME_LENGTH];
		for (int i = 0; i < GENOME_LENGTH; i++) {
			if (i <= mid) {
				offspringGenes[i] = genes[i];
			} else {
				offspringGenes[i] = b.getGenes()[i];
			}
		}
		return Chromosome(offspringGenes);
	}

	void mutate() {
		for (int i = 0; i < GENOME_LENGTH; i++) {
			float multiplier = (0.2f * rand()) / RAND_MAX - 0.1;
			genes[i] *= multiplier;
			if (multiplier < -0.1 || multiplier > 0.1) {
				printf("Multiplier is wrong. %f", multiplier);
			}
		}
	}

	void print() {
		printf("Chromosome: ");
		for (int i = 0; i < GENOME_LENGTH; i++) {
			printf("%f ", genes[i]);
		}
		printf("\nFitness: %f\n", getFitnessValue());
	}

};

bool fitnessComparator(Chromosome &a, Chromosome &b) {
	return a.getFitnessValue() < b.getFitnessValue();
}

void startIteration(Chromosome population[]) {
	int num_parents = NUMBER_CHROMOSOMES * (1 - CROSSOVER_RATE);
	int num_offsprings = NUMBER_CHROMOSOMES - num_parents;

	std::sort(population, population + NUMBER_CHROMOSOMES, fitnessComparator);

	int offspringStartIndex = num_parents;

	for (int i = offspringStartIndex; i < NUMBER_CHROMOSOMES; i++) {
		int fatherIndex = rand() % num_parents;
		int motherIndex = rand() % num_parents;
		Chromosome male = population[fatherIndex];
		Chromosome female = population[motherIndex];
		Chromosome offspring = male.crossover(female);

		if (MUTATION_FACTOR >= ((float) rand()) / RAND_MAX) {
			offspring.mutate();
		}
		population[i] = offspring;
	}
}

Chromosome geneticAlgorithm(Chromosome population[]) {
	for (int z = 0; z < NUM_EPOCHS; z++) {
		startIteration(population);
	}
	return population[0];
}

int main() {
	srand(time(NULL));
	Chromosome population[NUMBER_CHROMOSOMES];
	printf("Random population generated.");

	Chromosome best = geneticAlgorithm(population);
	best.print();

	return 0;
}

