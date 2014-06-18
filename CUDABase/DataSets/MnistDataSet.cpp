/*
 * MnistDataSet.cpp
 *
 *  Created on: May 26, 2014
 *      Author: cve
 */

#include "MnistDataSet.h"

template<typename M>
MnistDataSet<M>::MnistDataSet() :
DataSet<M>::DataSet() {
	this->X_test = NULL;
	this->Y_test = NULL;

	this->nTestData = 0;
}

template<typename M>
MnistDataSet<M>::MnistDataSet(const int& nDim, const int& nData, const int& nTestData, const int& nClasses)
: DataSet<M>::DataSet(nDim, nData, nClasses) {
	this->nTestData = nTestData;
	this->X = tensor<float, M>(extents[nData][nDim]);
	this->Y = tensor<float, M>(extents[nData][1]);
	this->X_test = tensor<float, M>(extents[nTestData][nDim]);
	this->Y_test = tensor<float, M>(extents[nTestData][1]);
	createData();
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
tensor<float, M> MnistDataSet<M>::convertToBinary(const tensor<float, M>& X) {
	tensor<float, M> result = X.copy();
	result /= 255.f;
	apply_scalar_functor(result, result, SF_GEQ, 0.5);
	return result;
}

template<typename M>
tensor<float, M> MnistDataSet<M>::getX_binary() {
	return convertToBinary(this->X);
}

template<typename M>
tensor<float, M> MnistDataSet<M>::getX_test_binary() {
	return convertToBinary(this->X_test);
}

template<typename M>
void MnistDataSet<M>::createData() {
	std::string path = "/home/local/datasets/MNIST";
	ifstream ftraind((path + "/train-images.idx3-ubyte").c_str());
	ifstream ftrainl((path + "/train-labels.idx1-ubyte").c_str());
	ifstream ftestd ((path + "/t10k-images.idx3-ubyte").c_str());
	ifstream ftestl ((path + "/t10k-labels.idx1-ubyte").c_str());

	char buf[16];
	ftraind.read(buf,16); ftrainl.read(buf, 8);
	ftestd.read(buf,16); ftestl.read(buf, 8);
	tensor<unsigned char, host_memory_space> traind(extents[this->nData][this->nDim]);
	tensor<unsigned char, host_memory_space> trainl(extents[this->nData]);
	tensor<unsigned char, host_memory_space> testd(extents[nTestData][this->nDim]);
	tensor<unsigned char, host_memory_space> testl(extents[nTestData]);
	ftraind.read((char*)traind.ptr(), traind.size());
	assert(ftraind.good());
	ftrainl.read((char*)trainl.ptr(), trainl.size());
	assert(ftrainl.good());
	ftestd.read((char*)testd.ptr(), testd.size());
	assert(ftestd.good());
	ftestl.read((char*)testl.ptr(), testl.size());
	assert(ftestl.good());

	tensor<unsigned char, M> train_d(extents[this->nData][this->nDim]);
	tensor<unsigned char, M> train_l(extents[this->nData]);
	tensor<unsigned char, M> test_d(extents[nTestData][this->nDim]);
	tensor<unsigned char, M> test_l(extents[nTestData]);

	train_d = traind;
	train_l = trainl;
	test_d = testd;
	test_l = testl;

	// conversion to float:
	convert(this->X, train_d);
	convert(this->Y, train_l);
	convert(this->X_test, test_d);
	convert(this->Y_test, test_l);
}

template<typename M>
MnistDataSet<M>::~MnistDataSet() {

}

