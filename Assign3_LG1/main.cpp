/*
 * main.cpp
 *
 *  Created on: May 16, 2014
 *      Author: cve
 */

#include "src/DataSet.h"
#include "src/LogisticRegression.h"

#define MEMORY dev_memory_space
#define N_DIM 2
#define N_CLASSES 4
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


	tensor<float, dev_memory_space> covariance(extents[N_CLASSES][N_DIM][N_DIM]);
	covariance(0, 0, 0) = 1.f;
	covariance(0, 1, 1) = 1.f;
	covariance(0, 1, 0) = 0.f;
	covariance(0, 0, 1) = 0.f;
	covariance(1, 0, 0) = 1.f;
	covariance(1, 1, 1) = 1.f;
	covariance(1, 1, 0) = 0.f;
	covariance(1, 0, 1) = 0.f;
	covariance(2, 0, 0) = 1.f;
	covariance(2, 1, 1) = 1.f;
	covariance(2, 1, 0) = 0.f;
	covariance(2, 0, 1) = 0.f;
	covariance(3, 0, 0) = 1.f;
	covariance(3, 1, 1) = 1.f;
	covariance(3, 1, 0) = 0.f;
	covariance(3, 0, 1) = 0.f;

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

	LogisticRegression<MEMORY> reg(0.08, N_ITERATIONS, N_CLASSES, N_DIM);
	reg.fit(train.getData(), train.getLabels());

	test.getClassification(reg.predict(test.getData()));

	cout<<"W: "<<reg.getW()<<endl;
	cout<<"B: "<<reg.getB()<<endl;

	return 0;
}



