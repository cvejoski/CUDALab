/*
 * MnistDataSet.cpp
 *
 *  Created on: May 18, 2014
 *      Author: cve
 */

#include "MnistDataSet.h"

template<typename M>
MnistDataSet<M>::MnistDataSet() {
	this->n_classes = 0;
	this->n_trainData = 0;
	this->n_testData = 0;
	this->n_dim = 0;

	this->X_test = NULL;
	this->Y_test = NULL;
	this->X_train = NULL;
	this->Y_train = NULL;
}

template<typename M>
MnistDataSet<M>::MnistDataSet(int n_dim, int n_trainData, int n_testData, int n_classes) {
	this->n_classes = n_classes;
	this->n_trainData = n_trainData;
	this->n_testData = n_testData;
	this->n_dim = n_dim;

	this->X_train = tensor<float, M>(extents[n_trainData][n_dim]);
	this->Y_train = tensor<float, M>(extents[n_trainData][1]);
	this->X_test = tensor<float, M>(extents[n_testData][n_dim]);
	this->Y_test = tensor<float, M>(extents[n_testData][1]);
}

template<typename M>
void MnistDataSet<M>::read() {

	std::string path = "/home/local/datasets/MNIST";
	ifstream ftraind((path + "/train-images.idx3-ubyte").c_str());
	ifstream ftrainl((path + "/train-labels.idx1-ubyte").c_str());
	ifstream ftestd ((path + "/t10k-images.idx3-ubyte").c_str());
	ifstream ftestl ((path + "/t10k-labels.idx1-ubyte").c_str());

	char buf[16];
	ftraind.read(buf,16); ftrainl.read(buf, 8);
	ftestd.read(buf,16); ftestl.read(buf, 8);
	tensor<unsigned char, host_memory_space> traind(extents[n_trainData][n_dim]);
	tensor<unsigned char, host_memory_space> trainl(extents[n_trainData]);
	tensor<unsigned char, host_memory_space> testd(extents[n_testData][n_dim]);
	tensor<unsigned char, host_memory_space> testl(extents[n_testData]);
	ftraind.read((char*)traind.ptr(), traind.size());
	assert(ftraind.good());
	ftrainl.read((char*)trainl.ptr(), trainl.size());
	assert(ftrainl.good());
	ftestd.read((char*)testd.ptr(), testd.size());
	assert(ftestd.good());
	ftestl.read((char*)testl.ptr(), testl.size());
	assert(ftestl.good());

	tensor<unsigned char, M> train_d(extents[n_trainData][n_dim]);
	tensor<unsigned char, M> train_l(extents[n_trainData]);
	tensor<unsigned char, M> test_d(extents[n_testData][n_dim]);
	tensor<unsigned char, M> test_l(extents[n_testData]);

	train_d = traind;
	train_l = trainl;
	test_d = testd;
	test_l = testl;

	// conversion to float:
	convert(this->X_train, train_d);
	convert(this->Y_train, train_l);
	convert(this->X_test, test_d);
	convert(this->Y_test, test_l);

}

template<typename M>
int MnistDataSet<M>::missClass(const tensor<float, M>& Y_predict) {
	int result = 0;
	tensor<float, M> diff(this->Y_test.shape());
	apply_binary_functor(diff, Y_test, Y_predict, BF_EQ);
	result = Y_test.shape(0) - sum(diff);
	return result;
}

template<typename M>
tensor<float, M> MnistDataSet<M>::getX_train() {
	return this->X_train;
}

template<typename M>
tensor<float, M> MnistDataSet<M>::getY_train() {
	return this->Y_train;
}

template<typename M>
tensor<float, M> MnistDataSet<M>::getX_test() {
	return this->X_test;
}

template<typename M>
tensor<float, M> MnistDataSet<M>::getY_test() {
	return this->Y_test;
}

template<typename M>
MnistDataSet<M>::~MnistDataSet() {
	// TODO Auto-generated destructor stub
}

template class MnistDataSet<host_memory_space>;
template class MnistDataSet<dev_memory_space>;
