/*
 * main.cpp
 *
 *  Created on: May 19, 2014
 *      Author: cve
 */

#include "DataSets/GaussianDataSet.h"
#include "Algorithms/LogisticRegression.h"



int main() {


	tensor<float, host_memory_space> mean(extents[2][2]);
	mean[0] = 2.f;
	mean[1] = 1.f;
	mean[2] = 5.f;
	mean[3] = 1.f;
//	mean[4] = 2.f;
//	mean[5] = 4.f;
//	mean[6] = 5.f;
//	mean[7] = 4.f;



	tensor<float, host_memory_space> covariance(extents[2][2][2]);
	covariance(0, 0, 0) = 1.f;
	covariance(0, 1, 1) = 1.f;
	covariance(0, 1, 0) = 0.f;
	covariance(0, 0, 1) = 0.f;
	covariance(1, 0, 0) = 1.f;
	covariance(1, 1, 1) = 1.f;
	covariance(1, 1, 0) = 0.f;
	covariance(1, 0, 1) = 0.f;
//	covariance(2, 0, 0) = 1.f;
//	covariance(2, 1, 1) = 1.f;
//	covariance(2, 1, 0) = 0.f;
//	covariance(2, 0, 1) = 0.f;
//	covariance(3, 0, 0) = 1.f;
//	covariance(3, 1, 1) = 1.f;
//	covariance(3, 1, 0) = 0.f;
//	covariance(3, 0, 1) = 0.f;


	GaussianDataSet<host_memory_space> train(2, 20, 2, covariance, mean);
	train.printToScreen();

	LogisticRegression<host_memory_space> m;


	return 0;
}
