/*
 * main.cpp
 *
 *  Created on: May 19, 2014
 *      Author: cve
 */

#include "DataSets/LinearDataSet.h"
#include "Algorithms/KernelRegression.h"
#include "Algorithms/Kernels/LinearKernel.h"
#include "Algorithms/OnLineLinearRegression.h"
#include "DataSets/RegressionDataSetGaus.h"
#include "DataSets/RegressionSigmoidDataSet.h"
#include "Algorithms/Kernels/GaussianKernel.h"
#include "Algorithms/Kernels/SigmoidKernel.h"
#include "DataSets/XORDataSet.h"
#include "Algorithms/KernelClassification.h"

#define N_OUTPUTS 3
#define N_DIM 2
#define N_TRAINING 200
#define N_ITERATION 5000
#define N_TEST 1000
#define MEMORY dev_memory_space

int main() {

	initCUDA(0);
	initialize_mersenne_twister_seeds(0);
	XORDataSet<MEMORY> xorTrain(2, N_TRAINING, 0.8, true);
	xorTrain.printToFile("./Results/Gaussian_train.dat");

	XORDataSet<MEMORY> xorTest(2, N_TEST, 0.8, false);
	xorTest.printToFile("./Results/Gaussian_test.dat");

	KernelClassification<MEMORY> kernelClassification(0.006, 0.0, N_ITERATION, N_OUTPUTS, N_DIM, new GaussianKernel<MEMORY>(0.8));
	kernelClassification.fit(xorTrain.getData(), xorTrain.getLabels());

	kernelClassification.predict(xorTest.getData(),"./Results/Gaussian_predicted.dat");
//
//	XORDataSet<MEMORY> xorTrain(2, N_TRAINING, 0.8, true);
//	xorTrain.printToFile("./Results/XORtrainM.dat");
//
//	XORDataSet<MEMORY> xorTest(2, N_TEST, 0.8, true);
//	xorTest.printToFile("./Results/XORtestM.dat");


//	KernelClassification<MEMORY> kernelClassification(0.006, 0.0, N_ITERATION, N_OUTPUTS, N_DIM, new GaussianKernel<MEMORY>(0.8));
//	kernelClassification.fit(xorTrain.getData(), xorTrain.getLabels());
//
//	kernelClassification.predict(xorTest.getData(),"./Results/XOR_predictedM.dat");

//	KernelClassification<MEMORY> kernelClassification(0.06, 0.0, N_ITERATION, N_OUTPUTS, N_DIM, new SigmoidKernel<MEMORY>(0.5, -0.5));
//	kernelClassification.fit(xorTrain.getData(), xorTrain.getLabels());
//
//	kernelClassification.predict(xorTest.getData(),"./Results/XOR_predictedM1.dat");
	return 0;
}




