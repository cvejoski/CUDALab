/*
 * main.cpp
 *
 *  Created on: May 19, 2014
 *      Author: cve
 */

#include "DataSets/MnistDataSet.h"
#include "DataSets/XORDataSet.h"
#include "DataSets/CifarDataSet.h"


#include "Algorithms/MLPClassification.h"
#include "Algorithms/Kernels/TanhKernel.h"
#include "Algorithms/Kernels/GaussianKernel.h"
#include "Algorithms/Helper.h"
#include "Algorithms/CrossValidator.h"
#include "Algorithms/KernelClassification.h"
#include "Factoris/KernelClassificationFactory.h"

#include "Algorithms/LogisticRegression.h"
#include "Factoris/LorRegFactory.h"


#define N_OUTPUTS 20
#define N_DIM 3072
#define N_TRAINING 50000
#define N_ITERATION 2000
#define N_TEST 10000
#define MEMORY dev_memory_space

void MLP_Classification();
void mnistClassification();
void cifarLogisticClassification();

int main() {

	initCUDA(0);
	initialize_mersenne_twister_seeds(0);
	cifarLogisticClassification();
	//MLP_Classification();
	//mnistClassification();
	//gaussianClassificationM();
	return 0;
}

void cifarLogisticClassification() {

	CifarDataSet<MEMORY> cifar1(N_DIM, N_TRAINING, N_TEST, N_OUTPUTS, false);
	//LogisticRegression<MEMORY> lg(0.000005, 0.001, N_ITERATION, N_OUTPUTS, N_DIM);
	vector<float> l_rate;
	vector<float> r_rate;
	vector<float> sigma;

	for (int i = 1; i <= 1000000; i *= 10) {
		r_rate.push_back(1.0/i);
		l_rate.push_back(1.0/i);
		cout<<1.0/i<<endl;
	}
	LogRegFactory<MEMORY>* m;

	m = new LogRegFactory<MEMORY>();
	CrossValidator<MEMORY> cross(l_rate, r_rate, N_OUTPUTS, N_ITERATION, 10, m);
	cross.fit(cifar1.getData(), cifar1.getLabels());
	//cout<<cifar1.getLabels()<<endl;
	//lg.fit(cifar1.getData(), cifar1.getLabels());
	//cout<<lg.predictWithError(cifar1.getX_test(), cifar1.getY_test())<<endl;
	cout<<cross.predictWithError(cifar1.getX_test(), cifar1.getY_test());
	cross.printBestModel();
}

void MLP_Classification() {
	MnistDataSet<MEMORY> mnist(N_DIM, N_TRAINING, N_TEST, N_OUTPUTS);

	MLPClassification<MEMORY> mlpClassification(0.00005, 0.000001, N_ITERATION, N_OUTPUTS, N_DIM, 280, new TanhKernel<MEMORY>());
	mlpClassification.fit_batch(mnist.getData(), mnist.getLabels(), 100);

	cout<<mlpClassification.predictWithError(mnist.getX_test(), mnist.getY_test());

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
	KernelClassification<MEMORY> kernelClassification(0.4, 0.001, N_ITERATION, N_OUTPUTS, N_DIM, new GaussianKernel<MEMORY>(850));
	{
		Helper<MEMORY> helper;
		tensor<float, MEMORY> destinationX(extents[11500][784]);
		tensor<float, MEMORY> destinationY(extents[11500][1]);
		helper.getRandomSubgroup(destinationX, destinationY, mnist.getData(), mnist.getLabels());
		kernelClassification.fit(destinationX, destinationY);
		destinationX.dealloc();
		destinationY.dealloc();
	}

	//c.fit_kernel(destinationX, destinationY);
	cout<<"ERRORS TEST DATA: "<<kernelClassification.predictWithError(mnist.getX_test(), mnist.getY_test());
}
