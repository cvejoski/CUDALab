/*
 * main.cpp
 *
 *  Created on: May 19, 2014
 *      Author: cve
 */

#include "DataSets/MnistDataSet.h"
#include "DataSets/XORDataSet.h"

#include "Algorithms/MLPClassification.h"
#include "Algorithms/Kernels/TanhKernel.h"
#include "Algorithms/Kernels/GaussianKernel.h"
#include "Algorithms/Helper.h"
#include "Algorithms/CrossValidator.h"
#include "Algorithms/KernelClassification.h"
#include "Factoris/KernelClassificationFactory.h"

#define N_OUTPUTS 10
#define N_DIM 784
#define N_TRAINING 60000
#define N_ITERATION 7000
#define N_TEST 10000
#define MEMORY dev_memory_space

void MLP_Classification();
void mnistClassification();
void gaussianClassificationM();

int main() {

	initCUDA(0);
	initialize_mersenne_twister_seeds(0);
	//MLP_Classification();
	mnistClassification();
	//gaussianClassificationM();
	return 0;
}

void MLP_Classification() {
	XORDataSet<MEMORY> xorTrain(2, N_TRAINING, 0.8, true);
	xorTrain.printToFile("./Results/XORtrainM.dat");

	XORDataSet<MEMORY> xorTest(2, N_TEST, 0.8, true);
	xorTest.printToFile("./Results/XORtestM.dat");

	MLPClassification<MEMORY> mlpClassification(0.3, 0.1, N_ITERATION, N_OUTPUTS, N_DIM, 50, new TanhKernel<MEMORY>());
	mlpClassification.fit(xorTrain.getData(), xorTrain.getLabels());

	cout<<mlpClassification.predictWithError(xorTest.getData(), xorTest.getLabels());
	mlpClassification.predict(xorTest.getData(),"./Results/MLP_predictedM.dat");
}

void mnistClassification() {
	vector<float> l_rate;
	vector<float> r_rate;
	vector<float> sigma;

	for (int i = 1; i <= 1000000; i *= 10) {
		sigma.push_back(1.0/i);
		cout<<1.0/i<<endl;
	}
	l_rate.push_back(0.1);
	r_rate.push_back(0.1);
	l_rate.push_back(0.01);
	r_rate.push_back(0.01);
	l_rate.push_back(0.001);
	r_rate.push_back(0.001);
	MnistDataSet<MEMORY> mnist(784, 60000, 10000, 10);

	//CrossValidator<MEMORY> c(l_rate, r_rate, sigma, N_OUTPUTS, N_ITERATION, 4, new KernelClassificationFactory<MEMORY>());
	KernelClassification<MEMORY> kernelClassification(0.4, 0.001, N_ITERATION, N_OUTPUTS, N_DIM, new GaussianKernel<MEMORY>(980));
	{
		Helper<MEMORY> helper;
		tensor<float, MEMORY> destinationX(extents[6500][784]);
		tensor<float, MEMORY> destinationY(extents[6500][1]);
		helper.getRandomSubgroup(destinationX, destinationY, mnist.getData(), mnist.getLabels());
		kernelClassification.fit(destinationX, destinationY);
		destinationX.dealloc();
		destinationY.dealloc();
	}

	//c.fit_kernel(destinationX, destinationY);
	cout<<"ERRORS TEST DATA: "<<kernelClassification.predictWithError(mnist.getX_test(), mnist.getY_test());
}
