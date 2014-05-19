/*
 * MnistDataSet.h
 *
 *  Created on: May 18, 2014
 *      Author: cve
 */

#ifndef MNISTDATASET_H_
#define MNISTDATASET_H_

#include <cuv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cuv;

template<typename M>
class MnistDataSet {
private:
	int n_dim;
	int n_trainData;
	int n_testData;
	int n_classes;

	tensor<float, M> X_train;
	tensor<float, M> Y_train;
	tensor<float, M> X_test;
	tensor<float, M> Y_test;
public:
	MnistDataSet();
	MnistDataSet(int n_dim, int n_trainData, int n_testData, int n_classes);
	void read();
	int missClass(const tensor<float, M>& Y_predict);
	tensor<float, M> getX_train();
	tensor<float, M> getY_train();
	tensor<float, M> getX_test();
	tensor<float, M> getY_test();
	virtual ~MnistDataSet();
};

#endif /* MNISTDATASET_H_ */
