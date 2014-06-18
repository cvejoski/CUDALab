/*
 * main.cpp
 *
 *  Created on: May 19, 2014
 *      Author: cve
 */

#include <DataSets/MnistDataSet.h>

#include <Factoris/LorRegFactory.h>
#include "src/CrossValidator.h"
#include "src/CrossValidator.cpp"

#define MEMORY dev_memory_space
#define N_DIM 784
#define N_CLASSES 10
#define N_ITERATIONS 100

int main() {

		initCUDA(0);
//		tensor<float, MEMORY> mean(extents[N_CLASSES][N_DIM]);
//		mean[0] = 2.f;
//		mean[1] = 1.f;
//		mean[2] = 5.f;
//		mean[3] = 1.f;
//	//	mean[4] = 2.f;
//	//	mean[5] = 4.f;
//	//	mean[6] = 5.f;
//	//	mean[7] = 4.f;
//
//
//		tensor<float, MEMORY> covariance(extents[N_CLASSES][N_DIM][N_DIM]);
//		covariance(0, 0, 0) = 1.f;
//		covariance(0, 1, 1) = 1.f;
//		covariance(0, 1, 0) = 0.f;
//		covariance(0, 0, 1) = 0.f;
//		covariance(1, 0, 0) = 1.f;
//		covariance(1, 1, 1) = 1.f;
//		covariance(1, 1, 0) = 0.f;
//		covariance(1, 0, 1) = 0.f;
//	//	covariance(2, 0, 0) = 1.f;
//	//	covariance(2, 1, 1) = 1.f;
//	//	covariance(2, 1, 0) = 0.f;
//	//	covariance(2, 0, 1) = 0.f;
//	//	covariance(3, 0, 0) = 1.f;
//	//	covariance(3, 1, 1) = 1.f;
//	//	covariance(3, 1, 0) = 0.f;
//	//	covariance(3, 0, 1) = 0.f;

	vector<float> l_rate;
	vector<float> r_rate;
	for (int i = 1; i <= 1000000; i*=10) {
		l_rate.push_back(1.0/i);
		r_rate.push_back(1.0/i);
		cout<<1.0/i<<endl;
	}

	MnistDataSet<MEMORY> mnist(784, 60000, 10000, 5);


//	GaussianDataSet<MEMORY> train(N_DIM, 300, N_CLASSES, covariance, mean);
//	GaussianDataSet<MEMORY> test(N_DIM, 2000, N_CLASSES, covariance, mean);
//	train.printToScreen();
//	train.printToFile("./Results/train.dat");
//	LogisticRegression<MEMORY>  logReg(0.01, 0, N_ITERATIONS, N_CLASSES, N_DIM);
//	logReg.fit(train.getData(), train.getLabels());


	LogRegFactory<MEMORY>* m;

	m = new LogRegFactory<MEMORY>();
	CrossValidator<MEMORY> c(l_rate, r_rate, N_CLASSES, N_ITERATIONS, 10, m);
	c.fit(mnist.getData(), mnist.getLabels());

	cout<<"ERRORS TEST DATA: "<<c.predictWithError(mnist.getX_test(), mnist.getY_test());
	c.printBestModel();
	return 0;
}


