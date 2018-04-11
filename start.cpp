#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <random>
#include <math.h>
#include <iostream>
using namespace std;
typedef long double HighlyPrecise;

const int GENOME_LENGTH = 16;
const int NUMBER_CHROMOSOMES = 320;

const double GENE_MIN = -1;
const double GENE_MAX = +1;

const float MUTATION_FACTOR = 0.2;
const float CROSSOVER_RATE = 0.6;

const int NUM_EPOCHS = 1000;

default_random_engine generator;
uniform_real_distribution<HighlyPrecise> geneValueDistribution(GENE_MIN, GENE_MAX);

class Chromosome {
	HighlyPrecise genes[GENOME_LENGTH];

public:

	Chromosome() {
		for (int j = 0; j < GENOME_LENGTH; j++) {
			genes[j] = geneValueDistribution(generator);
		}
	}

	Chromosome(HighlyPrecise genes[]) {
		memcpy(this->genes, genes, GENOME_LENGTH * sizeof(HighlyPrecise));
	}

	HighlyPrecise* getGenes() {
		return genes;
	}

	/**
	 * The current fitness function is: Summation (x * x). It needs to be minimized.
	 */
	HighlyPrecise getFitnessValue() {
		HighlyPrecise value = 0.0;
		for (int i = 0; i < GENOME_LENGTH; i++) {
			value += genes[i] * genes[i];
		}
		return value;
	}

	/**
	 * Takes two chromosomes and crosses the caller with the arguments and returns the offspring
	 * as a result.
	 */
	Chromosome crossover(Chromosome b) {
		int mid = GENOME_LENGTH / 2;
		HighlyPrecise offspringGenes[GENOME_LENGTH];
		for (int i = 0; i < GENOME_LENGTH; i++) {
			if (i <= mid) {
				offspringGenes[i] = genes[i];
			} else {
				offspringGenes[i] = b.getGenes()[i];
			}
		}
		Chromosome offspring(offspringGenes);
//		if(offspring.getFitnessValue() == getFitnessValue() || offspring.getFitnessValue() == b.getFitnessValue()) {
//			printf("Baccha on maa baap\n");
//		} else {
//			printf("NOT on maa baap\n");
//		}
		return Chromosome(offspringGenes);
	}

	void mutate() {
		for (int i = 0; i < GENOME_LENGTH; i++) {
			/* Random Multiplier between [-0.1 to 0.1) */
			HighlyPrecise multiplier = geneValueDistribution(generator) / 10;
			genes[i] *= multiplier;
			if (multiplier < -0.1 || multiplier > 0.1) {
				printf("Multiplier is wrong. %Le", multiplier);
			}
		}
	}

	/**
	 * Prints the chromosome and its fitness function.
	 */
	void print() {
		printf("Chromosome: ");
		for (int i = 0; i < GENOME_LENGTH; i++) {
			printf("%Le ", genes[i]);
		}
		printf("\nFitness: %Le\n", getFitnessValue());
	}

};

bool fitnessComparator(Chromosome &a, Chromosome &b) {
	return a.getFitnessValue() < b.getFitnessValue();
}

void startIteration(Chromosome population[]) {
	int num_parents = (1.0 * NUMBER_CHROMOSOMES * (1 - CROSSOVER_RATE));
	int num_offsprings = NUMBER_CHROMOSOMES - num_parents;

	uniform_int_distribution<int> parentIndexDistribution(0, num_parents);

	std::sort(population, population + NUMBER_CHROMOSOMES, fitnessComparator);

	int offspringStartIndex = num_parents;

	for (int i = offspringStartIndex; i < NUMBER_CHROMOSOMES; i++) {
		int fatherIndex = parentIndexDistribution(generator);
		int motherIndex = parentIndexDistribution(generator);
		if (motherIndex == fatherIndex) {
			// Should we? Because if we won't, there will be duplicate chromosomes.
			// TODO: Second opinion needed.
			continue;
		}

		Chromosome male = population[fatherIndex];
		Chromosome female = population[motherIndex];
		Chromosome offspring = male.crossover(female);

		// if mutation factor is greater than a random number from (0.0 to 1.0).
		if (MUTATION_FACTOR >= abs(geneValueDistribution(generator))) {
			offspring.mutate();
		}
		population[i] = offspring;
	}
}

Chromosome geneticAlgorithm(Chromosome population[]) {
	for (int z = 0; z < NUM_EPOCHS; z++) {
		startIteration(population);
	}
	std::sort(population, population + NUMBER_CHROMOSOMES, fitnessComparator);
	return population[0];
}

int main() {
	Chromosome population[NUMBER_CHROMOSOMES];
	// Start genetic algorithm.
	Chromosome best = geneticAlgorithm(population);
	best.print();
	return 0;
}

