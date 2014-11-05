/*
 * main.cpp
 *
 *  Created on: Jun 2, 2014
 *      Author: cve
 */
#include <time.h>

#include <DataSets/MnistDataSet.h>
#include <Algorithms/NaiveBayes.h>
#include <Algorithms/LogisticRegression.h>

#define MEMORY dev_memory_space
#define N_DIM 784
#define N_CLASSES 10
#define N_ITERATIONS 4500

int main() {
	initCUDA(0);
	//tensor<float, MEMORY> k(extents[60000][10000]);
	MnistDataSet<MEMORY> mnist(784, 60000, 10000, 10);
	NaiveBayes<dev_memory_space> bayes(10, 784, 0.0000000001);
	double time = 0.0;

	clock_t start = clock();
	bayes.fit(mnist.getX_binary(), mnist.getLabels());
	time += (double)(clock()-start)/CLOCKS_PER_SEC;
			cout<<"TIME: "<<time<<endl;

	cout<<"error:"<<bayes.predictWithError(mnist.getX_test_binary(), mnist.getY_test())<<endl;

////	bayes.plotParamAsImages();
	cout<<"end"<<endl;

	//LOGISTIC REGRESSION
	//initCUDA(1);
	//	MnistDataSet<MEMORY> mnist(784, 60000, 10000, 10);
	//
	//	LogisticRegression<MEMORY> lg(0.00001, 0.001, N_ITERATIONS, N_CLASSES, N_DIM);


	//	lg.fit(mnist.getData(), mnist.getLabels());

	//
	//
	//	lg.plotLearnedWeights();
	//	cout<<"Miss Classified # "<<lg.predictWithError(mnist.getX_test(), mnist.getY_test())<<endl;
	//	cout<<"end"<<endl;

	return 0;
}





