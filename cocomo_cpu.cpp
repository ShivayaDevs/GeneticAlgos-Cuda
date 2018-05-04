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

long double project[20], project_tdev[20];

default_random_engine generator;
uniform_real_distribution<HighlyPrecise> geneValueDistribution(GENE_MIN, GENE_MAX);
uniform_real_distribution<HighlyPrecise> geneValueDistribution2(0, 1);

class Chromosome {
	HighlyPrecise genes[GENOME_LENGTH];

public:

	Chromosome() {
		cout<<"\n";
		for (int j = 0; j < GENOME_LENGTH; j++) {
			genes[j] = geneValueDistribution(generator);
			cout<<genes[j]<<" ";
		}
	}

	Chromosome(HighlyPrecise genes[]) {
		memcpy(this->genes, genes, GENOME_LENGTH * sizeof(HighlyPrecise));
	}

	HighlyPrecise* getGenes() {
		return genes;
	}

	// long double dev_time(int j){
	// HighlyPrecise effort = genes[0] * pow((project[j]), genes[1]);
	// 	HighlyPrecise tdev = genes[2] * pow(effort, genes[3]);
	// 	return tdev;
	// }

    long double get_effort(int j){
        HighlyPrecise effort = genes[0] * pow((project[j]), genes[1]);
        return effort;
    }


	/**
	 * The current fitness function is: Summation (x * x). It needs to be minimized.
	 */
	HighlyPrecise getFitnessValue() {
		HighlyPrecise value = 0.0;


		// for(int j=0; j<20; j++){
		// 	value += fabs((project_tdev[j] - dev_time(j)))/ (double) project_tdev[j];
		// }
		// for (int i = 0; i < GENOME_LENGTH; i++) {
		// 	value += genes[i] * genes[i];
		// }
        for(int j=0; j<12; j++){
            value += fabs(project_tdev[j] - get_effort(j))*fabs(project_tdev[j] - get_effort(j));
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
			HighlyPrecise multiplier = geneValueDistribution2(generator);
			genes[i] *= multiplier;
		}
	}

	/**
	 * Prints the chromosome and its fitness function.
	 */
	void print() {
		// printf("Chromosome: ");
		// for (int i = 0; i < GENOME_LENGTH; i++) {
		// 	printf("%Le ", genes[i]);
		// }
		// printf("\nFitness: %Le\n", getFitnessValue());

        for (int j=0; j<12; j++){
            cout<<"Project["<<j<<"] "<<project_tdev[j]<<" : effort = "<<get_effort(j)<<"\n";
        }

        cout<<"A: "<<genes[0]<<"\n";
        cout<<"B: "<<genes[1]<<"\n";
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
	

	// project[0] = 10; project_tdev[0] = 13;
	// project[1] = 17; project_tdev[1] = 12;
	// project[2] = 10; project_tdev[2] = 15;
	// project[3] = 24; project_tdev[3] = 18;
	// project[4] = 19; project_tdev[4] = 13;
	// project[5] = 19; project_tdev[5] = 14;
	// project[6] = 10; project_tdev[6] = 15;
	// project[7] = 15; project_tdev[7] = 13;
	// project[8] = 10; project_tdev[8] = 12;
	// project[9] = 10; project_tdev[9] = 12;
	// project[10] = 17; project_tdev[10] = 22;
	// project[11] = 11; project_tdev[11] = 19;
	// project[12] = 15; project_tdev[12] = 18;
	// project[13] = 15; project_tdev[13] = 19;
	// project[14] = 13; project_tdev[14] = 21;
	// project[15] = 14; project_tdev[15] = 20;
	// project[16] = 15; project_tdev[16] = 19;
	// project[17] = 15; project_tdev[17] = 20;
	// project[18] = 13; project_tdev[18] = 15;
	// project[19] = 18; project_tdev[19] = 19;

     project[0] = 46.2; project_tdev[0] = 96.0;
    project[1] = 46.5; project_tdev[1] = 79.0;
    project[2] = 31.1; project_tdev[2] = 39.6;
    project[3] = 12.8; project_tdev[3] = 18.9;
    project[4] = 10.5; project_tdev[4] = 10.3;
    project[5] = 21.5; project_tdev[5] = 28.5;
    project[6] = 3.1; project_tdev[6] = 7.0;
    project[7] = 4.2; project_tdev[7] = 9.0;
    project[8] = 7.8; project_tdev[8] = 7.3;
    project[9] = 2.1; project_tdev[9] = 5.0;
    project[10] = 5.0; project_tdev[10] = 8.4;
    project[11] = 9.7; project_tdev[11] = 15.6;
    project[12] = 12.5; project_tdev[12] = 23.9;
    // project[13] = 15; project_tdev[13] = 19;
    // project[14] = 13; project_tdev[14] = 21;
    // project[15] = 14; project_tdev[15] = 20;
    // project[16] = 15; project_tdev[16] = 19;
    // project[17] = 15; project_tdev[17] = 20;
    // project[18] = 13; project_tdev[18] = 15;
    // project[19] = 18; project_tdev[19] = 19;





	Chromosome population[NUMBER_CHROMOSOMES];
	// Start genetic algorithm.
	Chromosome best = geneticAlgorithm(population);
	best.print();
	return 0;
}

