#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <random>
#include <math.h>
#include <iostream>
using namespace std;
typedef long double HighlyPrecise;

const int GENOME_LENGTH = 2;
const int NUMBER_CHROMOSOMES = 100;

const double GENE_MIN = 0.0;
const double GENE_MAX = 10.0;

const float MUTATION_FACTOR = 0.8;
const float CROSSOVER_RATE = 0.4;

const int NUM_EPOCHS = 1500;

const int NUM_PROJECTS = 12;

HighlyPrecise kiloLinesOfCode[NUM_PROJECTS];
HighlyPrecise actualEfforts[NUM_PROJECTS];

default_random_engine generator;
uniform_real_distribution<HighlyPrecise> geneValueDistribution(GENE_MIN, GENE_MAX);
uniform_real_distribution<HighlyPrecise> geneValueDistribution2(0, 1);

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

	HighlyPrecise getEstimatedEffort(int projectId) {
		// Effort is estimated by cocomo param (a * KLOC^b).
		return genes[0] * pow(kiloLinesOfCode[projectId], genes[1]);
	}

	HighlyPrecise getFitnessValue() {
		HighlyPrecise value = 0.0;
		for (int i = 0; i < NUM_PROJECTS; i++) {
			// TODO: See the possibility of a square here.
			value += fabs(actualEfforts[i] - getEstimatedEffort(i))
					* fabs(actualEfforts[i] - getEstimatedEffort(i));
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
		return Chromosome(offspringGenes);
	}

	void mutate() {
		for (int i = 0; i < GENOME_LENGTH; i++) {
			/* Random Multiplier between [-0.1 to 0.1) */
			HighlyPrecise multiplier = geneValueDistribution2(generator);
			genes[i] *= multiplier;
		}
	}

	/**
	 * Prints the chromosome and its fitness function.
	 */
	void print() {
		printf("==> Results:\nProjectId\tKloc\tActualEffort\tEstimatedEffortGA\n");
		for (int i = 0; i < NUM_PROJECTS; i++) {
			printf("%d\t%Lf\t%Lf\t%Lf\n", i, kiloLinesOfCode[i], actualEfforts[i],
					getEstimatedEffort(i));
		}
		printf("==> Parameters: ");
		cout << "A: " << genes[0] << "\n";
		cout << "B: " << genes[1] << "\n";
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
		if (MUTATION_FACTOR >= abs(geneValueDistribution2(generator))) {
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

	FILE *fp = fopen("cocomo.dataset", "r");
	if (fp == NULL) {
		printf("ERROR: Unable to open dataset file.");
		return 1;
	}
	printf("==> Training data:\nKLOC\t\tActualEffort\n");
	for (int i = 0; i < NUM_PROJECTS && !feof(fp); i++) {
		fscanf(fp, "%Lf %Lf", &kiloLinesOfCode[i], &actualEfforts[i]);
		printf("%Lf\t%Lf\n", kiloLinesOfCode[i], actualEfforts[i]);
	}
	fclose(fp);

	Chromosome population[NUMBER_CHROMOSOMES];
	// Start genetic algorithm.
	Chromosome best = geneticAlgorithm(population);
	best.print();
	return 0;
}

