/*
 * DataSet.cpp
 *
 *  Created on: May 2, 2014
 *      Author: cve
 */

#include "DataSet.h"
#include <iostream>

using namespace std;

template<typename M>
DataSet<M>::DataSet() {
	this->n_dim = 0;
	this->n_data = 0;
	this->n_outputs = 0;

	this->X = NULL;
	this->Y = NULL;
	this->w = NULL;
	this->b = NULL;
}

template<typename M>
DataSet<M>::DataSet(int n_dim, int n_data, int n_outputs, tensor<float, M> w,
		tensor<float, M> b) {

	this->n_dim = n_dim;
	this->n_data = n_data;
	this->n_outputs = n_outputs;

	this->X = new tensor<float, M>(extents[n_data][n_dim]);
	this->Y = new tensor<float, M>(extents[n_data][n_outputs]);
	this->w = w;
	this->b = b;
}

template<typename M>
void DataSet<M>::createData() {
	initialize_mersenne_twister_seeds(0);
	tensor<float, M> error((*Y).shape());
	error = 0.f;
	add_rnd_normal(error);
	error *= .05f;
	fill_rnd_uniform(*X);
	prod(*Y, *X, w, 'n', 'n', 1.f, 0.f);
	matrix_plus_row(*Y, b);
	(*Y) += error;
}

template<typename M>
tensor<float, M> DataSet<M>::getData() {
	return *X;
}

template<typename M>
tensor<float, M> DataSet<M>::getLabels() {
	return *Y;
}

template<typename M>
tensor<float, M> DataSet<M>::getW() {
	return w;
}

template<typename M>
tensor<float, M> DataSet<M>::getBias() {
	return b;
}

template<typename M>
DataSet<M>::~DataSet() {
	// TODO Auto-generated destructor stub
}

template class DataSet<host_memory_space>;
template class DataSet<dev_memory_space>;
