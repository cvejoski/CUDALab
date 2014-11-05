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
#include "Factoris/MlpFactory.h"


#define N_OUTPUTS 20
#define N_DIM 3072
#define N_TRAINING 50000
#define N_ITERATION 300
#define N_TEST 10000
#define MEMORY dev_memory_space

void MLP_Classification();
void mnistClassification();
void cifarLogisticClassification();
void MLPCifar_Classification();

int main() {

	initCUDA(0);
	initialize_mersenne_twister_seeds(0);
	//MLPCifar_Classification();
	//cifarLogisticClassification();
	MLP_Classification();
	//mnistClassification();
	//gaussianClassificationM();
	return 0;
}

void cifarLogisticClassification() {
	CifarDataSet<MEMORY> cifar1(N_DIM, N_TRAINING, N_TEST, N_OUTPUTS, false);
	vector<float> l_rate;
	vector<float> r_rate;
	vector<float> sigma;

//	for (int i = 10; i <= 10000000; i *= 10) {
//		r_rate.push_back(1.0/i);
//		l_rate.push_back(1.0/i);
//		cout<<1.0/i<<endl;
//	}

	r_rate.push_back(0.0001);
	l_rate.push_back(0.0000001);
	LogRegFactory<MEMORY>* m;

	double time = 0.0;

	m = new LogRegFactory<MEMORY>();
	CrossValidator<MEMORY> cross(l_rate, r_rate, N_OUTPUTS, N_ITERATION, 10, m);
	clock_t start = clock();
	cross.fit(cifar1.getData(), cifar1.getLabels());
	time += (double)(clock()-start)/CLOCKS_PER_SEC;
	cout<<"TIME: "<<time<<endl;
	cout<<"TEST DATA"<<cross.predictWithError(cifar1.getX_test(), cifar1.getY_test())<<endl;

//	LogisticRegression<MEMORY> l(0.000001, 0.001, N_ITERATION, N_OUTPUTS, N_DIM);
//	l.fit(cifar1.getData(), cifar1.getLabels());
//	cout<<"TEST DATA"<<l.predictWithError(cifar1.getX_test(), cifar1.getY_test())<<endl;
//	//cross.printBestModel();
}

void MLP_Classification() {

	CifarDataSet<MEMORY> cifar1(N_DIM, N_TRAINING, N_TEST, N_OUTPUTS, false);

	MLPClassification<MEMORY> mlpClassification(0.001, 0.0000001, N_ITERATION, N_OUTPUTS, N_DIM, 3000, new TanhKernel<MEMORY>());
	mlpClassification.fit_batch(cifar1.getData(), cifar1.getLabels(), cifar1.getX_test(), cifar1.getY_test(), 30);

	cout<<mlpClassification.predictWithError(cifar1.getX_test(), cifar1.getY_test());
	mlpClassification.confusionMatrix(cifar1.getX_test(), cifar1.getY_test());

}

void MLPCifar_Classification() {
	CifarDataSet<MEMORY> cifar1(N_DIM, N_TRAINING, N_TEST, N_OUTPUTS, false);
	vector<float> l_rate;
	vector<float> r_rate;
	vector<float> sigma;

//	for (int i = 100; i <= 10000000; i *= 10) {
//		r_rate.push_back(1.0/i);
//
//		cout<<1.0/i<<endl;
//	}
	r_rate.push_back(0.00000001);
	l_rate.push_back(0.001);


	sigma.push_back(1500);
	MlpFactory<MEMORY>* m;

	m = new MlpFactory<MEMORY>();
	CrossValidator<MEMORY> cross(l_rate, r_rate, sigma, N_OUTPUTS, N_ITERATION, 10, m);
	cross.fit_kernel(cifar1.getData(), cifar1.getLabels());
	cout<<"TEST DATA"<<cross.predictWithError(cifar1.getX_test(), cifar1.getY_test())<<endl;


//	MLPClassification<MEMORY> mlpClassification(0.00005, 0.0000001, N_ITERATION, N_OUTPUTS, N_DIM, 880, new TanhKernel<MEMORY>());
//	mlpClassification.fit_batch(mnist.getData(), mnist.getLabels(), 20);
//
//	cout<<mlpClassification.predictWithError(mnist.getX_test(), mnist.getY_test());

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
