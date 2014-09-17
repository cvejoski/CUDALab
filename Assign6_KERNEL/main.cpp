/*
 * main.cpp
 *
 *  Created on: May 19, 2014
 *      Author: cve
 */

#include <DataSets/LinearDataSet.h>
#include <DataSets/RegressionDataSetGaus.h>
#include <DataSets/RegressionSigmoidDataSet.h>
#include <DataSets/XORDataSet.h>
#include <DataSets/MnistDataSet.h>

#include <Algorithms/Kernels/GaussianKernel.h>
#include <Algorithms/Kernels/LinearKernel.h>
#include <Algorithms/Kernels/SigmoidKernel.h>

#include <Algorithms/KernelClassification.h>
#include <Algorithms/KernelRegression.h>
#include <Algorithms/OnLineLinearRegression.h>
#include <Algorithms/CrossValidator.h>
#include <Algorithms/Helper.h>

#include <Factoris/KernelClassificationFactory.h>

#define N_OUTPUTS 3
#define N_DIM 2
#define N_TRAINING 1000
#define N_ITERATION 5000
#define N_TEST 500
#define MEMORY dev_memory_space


void linearRegression();
void gaussianRegression();
void sigmoidRegression();
void gaussianClassification();
void sigmoidClassification();
void gaussianClassificationM();
void sigmoidClassificationM();
void mnistClassification();

int main() {

	initCUDA(0);
	initialize_mersenne_twister_seeds(0);
	//mnistClassification();
	 sigmoidClassificationM();
	return 0;
}

void linearRegression() {

	tensor<float, MEMORY> w_0(extents[N_DIM][N_OUTPUTS]);
	tensor<float, MEMORY> b_0(extents[N_OUTPUTS]);


	fill_rnd_uniform(w_0);
	fill_rnd_uniform(b_0);

	LinearDataSet<MEMORY> dsTrain(N_DIM, N_TRAINING, N_OUTPUTS, w_0, b_0);

	LinearDataSet<MEMORY> dsTest(N_DIM, N_TEST, N_OUTPUTS, w_0, b_0);
	dsTrain.printToFile("./Results/KernelRegression/a_train.dat");
	cout<<w_0<<endl;
	cout<<b_0<<endl;


//	KernelRegression<MEMORY> kr(0.006, 0.0001, N_ITERATION, N_OUTPUTS, N_DIM, new LinearKernel<MEMORY>());
//	kr.fit(dsTrain.getData(), dsTrain.getLabels());
//	kr.predict(dsTest.getData(), "./Results/KernelRegression/a_predicted.dat");

	OnLineLinearRegression<MEMORY> o(0.0005, 0.0001, N_ITERATION, N_OUTPUTS, N_DIM);

	o.fit(dsTrain.getData(), dsTrain.getLabels());
	o.predict(dsTest.getData());

	cout<<o.getW()<<endl;
	cout<<o.getB()<<endl;

}

void gaussianRegression() {
	RegressionDataSetGaus<MEMORY> dsTrain(N_DIM, N_TRAINING, N_OUTPUTS, 0.05, 100);
	dsTrain.printToFile("./Results/KernelRegression/Gaussian_train.dat");

	RegressionDataSetGaus<MEMORY> dsTest(N_DIM, N_TEST, N_OUTPUTS, 0.05, 100);
	dsTest.printToFile("./Results/KernelRegression/Gaussian_test.dat");

	KernelRegression<MEMORY> kr(0.006, 0.0001, N_ITERATION, N_OUTPUTS, N_DIM, new GaussianKernel<MEMORY>(0.05));
	kr.fit(dsTrain.getData(), dsTrain.getLabels());
	kr.predict(dsTest.getData(), "./Results/KernelRegression/Gaussian_predicted.dat");
}

void sigmoidRegression() {
	RegressionSigmoidDataSet<MEMORY> dsTrain(N_DIM, N_TRAINING, N_OUTPUTS, 2, -0.005, 150);
	dsTrain.printToFile("./Results/KernelRegression/Sigmoid_train.dat");

	RegressionSigmoidDataSet<MEMORY> dsTest(N_DIM, N_TEST, N_OUTPUTS, 2, -0.005, 150);
	dsTest.printToFile("./Results/KernelRegression/Sigmoid_test.dat");

	KernelRegression<MEMORY> kr(0.001, 0.001, N_ITERATION, N_OUTPUTS, N_DIM, new SigmoidKernel<MEMORY>(2, -0.005));
	kr.fit(dsTrain.getData(), dsTrain.getLabels());
	kr.predict(dsTest.getData(), "./Results/KernelRegression/Sigmoid_predicted.dat");
}

void gaussianClassification() {
	XORDataSet<MEMORY> xorTrain(2, N_TRAINING, 0.8, false);
	xorTrain.printToFile("./Results/KernelClassification/Gaussian_train.dat");

	XORDataSet<MEMORY> xorTest(2, N_TEST, 0.8, false);
	xorTest.printToFile("./Results/KernelClassification/Gaussian_test.dat");

	KernelClassification<MEMORY> kernelClassification(0.006, 0.0, N_ITERATION, N_OUTPUTS, N_DIM, new GaussianKernel<MEMORY>(0.8));
	kernelClassification.fit(xorTrain.getData(), xorTrain.getLabels());

	kernelClassification.predict(xorTest.getData(),"./Results/KernelClassification/Gaussian_predicted.dat");
}

void gaussianClassificationM() {
	XORDataSet<MEMORY> xorTrain(2, N_TRAINING, 0.8, true);
	xorTrain.printToFile("./Results/KernelClassification/Gaussian_trainM.dat");

	XORDataSet<MEMORY> xorTest(2, N_TEST, 0.8, true);
	xorTest.printToFile("./Results/KernelClassification/Gaussian_testM.dat");

	KernelClassification<MEMORY> kernelClassification(0.006, 0.0, N_ITERATION, N_OUTPUTS, N_DIM, new GaussianKernel<MEMORY>(0.8));
	kernelClassification.fit(xorTrain.getData(), xorTrain.getLabels());

	kernelClassification.predict(xorTest.getData(),"./Results/KernelClassification/Gaussian_predictedM.dat");
}

void sigmoidClassification() {
	XORDataSet<MEMORY> xorTrain(2, N_TRAINING, 0.8, false);
	xorTrain.printToFile("./Results/KernelClassification/Sigmoid_train.dat");

	XORDataSet<MEMORY> xorTest(2, N_TEST, 0.8, false);
	xorTest.printToFile("./Results/KernelClassification/Sigmoid_test.dat");

	KernelClassification<MEMORY> kernelClassification(0.06, 0.0, N_ITERATION, N_OUTPUTS, N_DIM, new SigmoidKernel<MEMORY>(0.5, -0.5));
	kernelClassification.fit(xorTrain.getData(), xorTrain.getLabels());

	kernelClassification.predict(xorTest.getData(),"./Results/KernelClassification/Sigmoid_predicted.dat");
}

void sigmoidClassificationM() {
	XORDataSet<MEMORY> xorTrain(2, N_TRAINING, 0.8, true);
	xorTrain.printToFile("./Results/KernelClassification/Sigmoid_trainM.dat");

	XORDataSet<MEMORY> xorTest(2, N_TEST, 0.8, true);
	xorTest.printToFile("./Results/KernelClassification/Sigmoid_testM.dat");

	KernelClassification<MEMORY> kernelClassification(0.06, 0.0, N_ITERATION, N_OUTPUTS, N_DIM, new SigmoidKernel<MEMORY>(0.5, -0.5));
	kernelClassification.fit(xorTrain.getData(), xorTrain.getLabels());

	kernelClassification.predict(xorTest.getData(),"./Results/KernelClassification/Sigmoid_predictedM.dat");
}

