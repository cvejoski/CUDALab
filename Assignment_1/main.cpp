/*
 * main.cpp
 *
 *  Created on: Apr 17, 2014
 *      Author: cve
 */

#include <iostream>
#include <cuv.hpp>
#include <time.h>

#include "src/NearestNeighbor.h"
#include "src/NearestNeighbor.cpp"

using namespace std;
using namespace cuv;

void performance_measure();
void pointsClass();
void minstClass();

int main() {
	//performance_measure();
	//pointsClass();
	minstClass();
	cout<<"END\n";
	return 0;
}

void performance_measure() {

	int i = 2;
	NearestNeighbor<float, host_memory_space> performance;

	do {
			double time = 0.0;
			performance.initialize(i, 4, 50, 25);
			performance.createTrainDS();
			performance.createTestDS();
			for (int j = 0; j<=50; j++) {
				clock_t start = clock();
				performance.calcDistance();
				time += (double)(clock()-start)/CLOCKS_PER_SEC;
			}
			cout<<i<<" "<<time/50<<endl;
			performance.~NearestNeighbor();

		(i<100) ? i++ : i+=500;
	} while (i<=10000);
}

void pointsClass() {
	NearestNeighbor<float, dev_memory_space> n;
	n.initialize(2, 4, 500, 100);
	n.createTrainDS();
	n.createTestDS();
	n.calcDistance();
}

void minstClass() {
		NearestNeighbor<float, dev_memory_space> t;
		t.initializeMINST(784, 60000, 10000);
		t.readMINST();
		cout<<"CALCULATING DISTANCE\n";
		t.classifiedMINST();
		t.exportMINST();
}
