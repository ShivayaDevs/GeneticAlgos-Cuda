#include <stdio.h>
#include <string.h>
//#include <cstdlib>
#include <time.h>
#include <algorithm>
#include <random>
#include <math.h>
#include <iostream>
using namespace std;
typedef long double genetype;

const int GENOME_LENGTH = 16;
const int NUMBER_CHROMOSOMES = 100;

const double GENE_MIN = -1;
const double GENE_MAX = +1;

const float MUTATION_FACTOR = 0.2;
const float CROSSOVER_RATE = 0.6;

const int NUM_EPOCHS = 1000;

default_random_engine generator;
uniform_real_distribution<genetype> geneValueDistribution(GENE_MIN, GENE_MAX);

class Chromosome {
	genetype genes[GENOME_LENGTH];

public:

	Chromosome() {
		for (int j = 0; j < GENOME_LENGTH; j++) {
			genes[j] = geneValueDistribution(generator);
		}
	}

	Chromosome(genetype genes[]) {
		memcpy(this->genes, genes, GENOME_LENGTH * sizeof(genetype));
	}

	genetype* getGenes() {
		return genes;
	}

	genetype getFitnessValue() {
		genetype value = 0.0;
		for (int i = 0; i < GENOME_LENGTH; i++) {
			value += genes[i] * genes[i];
		}
//		printf("FITNESS: %e", value);
		return value;
	}

	Chromosome crossover(Chromosome b) {
//		printf("CROSSING OVER...\n");
		int mid = GENOME_LENGTH / 2;
		genetype offspringGenes[GENOME_LENGTH];
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
//		printf("MUTATING...\n");
		for (int i = 0; i < GENOME_LENGTH; i++) {
			genetype multiplier = geneValueDistribution(generator) / 10;
			genes[i] *= multiplier;
			if (multiplier < -0.1 || multiplier > 0.1) {
				printf("Multiplier is wrong. %f", multiplier);
			}
		}
	}

	void print() {
		printf("Chromosome: ");
		for (int i = 0; i < GENOME_LENGTH; i++) {
			printf("%e ", genes[i]);
		}
		printf("\nFitness: %e\n", getFitnessValue());
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
//	printf("Num_parents:%d , NumChromosomes:%d", num_parents, NUMBER_CHROMOSOMES);
	int count = 0;

	for (int i = offspringStartIndex; i < NUMBER_CHROMOSOMES; i++) {
		int fatherIndex = parentIndexDistribution(generator);
		int motherIndex = parentIndexDistribution(generator);


		if(motherIndex == fatherIndex) {
			count++;
			continue;
		}

		Chromosome male = population[fatherIndex];
		Chromosome female = population[motherIndex];
		Chromosome offspring = male.crossover(female);

		if (MUTATION_FACTOR >= abs(geneValueDistribution(generator))) {
			offspring.mutate();
		}
		population[i] = offspring;
	}
//	printf("Saved you : %d times\n", count);
}

Chromosome geneticAlgorithm(Chromosome population[]) {
	for (int z = 0; z < NUM_EPOCHS; z++) {
		printf("Iteration:%d\n", z);
		startIteration(population);
	}
	std::sort(population, population + NUMBER_CHROMOSOMES, fitnessComparator);
	return population[0];
}

int main() {
//	srand(time(NULL));
	Chromosome population[NUMBER_CHROMOSOMES];

	Chromosome best = geneticAlgorithm(population);
	best.print();

	return 0;
}

