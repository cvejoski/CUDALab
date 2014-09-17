/*
 * main.cpp
 *
 *  Created on: May 16, 2014
 *      Author: cve
 */

#include "src/DataSet.h"
#include "src/LogisticRegression.h"

#include "src/MnistDataSet.h"

#define MEMORY dev_memory_space
#define N_DIM 5
#define N_CLASSES 2
#define N_ITERATIONS 5000


int main() {
	tensor<float, dev_memory_space> mean(extents[N_CLASSES][N_DIM]);
	mean[0] = 2.f;
	mean[1] = 1.f;
	mean[2] = 5.f;
	mean[3] = 1.f;
	mean[4] = 2.f;
	mean[5] = 4.f;
	mean[6] = 5.f;
	mean[7] = 4.f;
	mean[8] = 4.f;
	mean[9] = 4.f;

	tensor<float, dev_memory_space> covariance(extents[N_CLASSES][N_DIM][N_DIM]);
	covariance = 0.f;
	covariance(0, 0, 0) = 1.f;
	covariance(0, 1, 1) = 1.f;
	covariance(0, 2, 2) = 1.f;
	covariance(0, 3, 3) = 1.f;
	covariance(0, 4, 4) = 1.f;


	covariance(1, 0, 0) = 1.f;
	covariance(1, 1, 1) = 1.f;
	covariance(1, 2, 2) = 1.f;
	covariance(1, 3, 3) = 1.f;
	covariance(1, 4, 4) = 1.f;

	DataSet<MEMORY> train(N_DIM, 240, N_CLASSES, covariance, mean);
	train.createData();
	train.printData();
	cout<<train.getLabels()<<endl;
	train.printData("./Results/train4R.dat");

	DataSet<MEMORY> test(N_DIM, 120, N_CLASSES, covariance, mean);
	test.createData();
	test.printData();
	cout<<test.getLabels()<<endl;
	test.printData("./Results/test4R.dat");

//	MnistDataSet<MEMORY> mnist(784, 60000, 10000, 10);
//	mnist.read();

	LogisticRegression<MEMORY> reg(0.6, 0.5, N_ITERATIONS, N_CLASSES, N_DIM);

	reg.fit(train.getData(), train.getLabels());
	//reg.fit(mnist.getX_train(), mnist.getY_train());

//	cout<<"Miss Classified # "<<mnist.missClass(reg.predict(mnist.getX_test()))<<endl;


	cout<<"W: "<<reg.getW()<<endl;
	cout<<"B: "<<reg.getB()<<endl;

	return 0;
}



