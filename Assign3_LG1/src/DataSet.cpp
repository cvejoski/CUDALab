/*
 * DataSet.cpp
 *
 *  Created on: May 16, 2014
 *      Author: cve
 */

#include "DataSet.h"

template<typename M>
DataSet<M>::DataSet() {
	this->n_dim = 0;
	this->n_data = 0;
	this->n_classes = 2;

	this->mean = NULL;
	this->covariance = NULL;
	this->X = NULL;
	this->Y = NULL;
}

template<typename M>
DataSet<M>::DataSet(int n_dim, int n_data, int n_classes, tensor<float, M> c,
		tensor<float, M> m) {
	this->n_dim = n_dim;
	this->n_data = n_data;
	this->n_classes = n_classes;

	this->mean = m;
	this->covariance = c;
	this->X = tensor<float, M>(extents[n_data][n_dim]);
	this->Y = tensor<float, M>(extents[n_data][1]);
}

template<typename M>
void DataSet<M>::createData() {
	initialize_mersenne_twister_seeds(time(NULL));
	X = 0.f;
	add_rnd_normal(X);
	int portion = n_data/n_classes;
	for (short int i = 0; i < n_classes; i++) {
		tensor_view<float, M> t_vC = covariance[indices[i]];
		tensor_view<float, M> t_vM = mean[indices[i]];
		tensor_view<float, M> t_vX = X[indices[index_range(portion*i, (i+1)*portion)]];
		tensor_view<float, M> t_vY = Y[indices[index_range(portion*i, (i+1)*portion)]];
		t_vY = i;
		tensor<float, M> result(t_vX.shape());
		prod(result, t_vX, t_vC);
		matrix_plus_row(result, t_vM);
		X[indices[index_range(portion*i, (i+1)*portion)]] = result;
	}
}

template<typename M>
tensor<float, M> DataSet<M>::getData() {
	return X;
}

template<typename M>
tensor<float, M> DataSet<M>::getLabels() {
	return this->Y;
}

template<typename M>
tensor<float, M> DataSet<M>::getCovariance() {
	return this->covariance;
}

template<typename M>
tensor<float, M> DataSet<M>::getMean() {
	return this->mean;
}

template<typename M>
void DataSet<M>::printData(){
	tensor<float, dev_memory_space> tmp = X;
	int portion = n_data/n_classes;
	for (int i = 0; i < this->n_classes; i++){
		cout<<"CLASS "<<i<<":\n";
		for (int j = i*portion; j < portion*(i+1); j++) {
			for (int k = 0; k < this->n_dim; k++)
				cout<<X(j, k)<<" ";
			cout<<endl;
		}
	}
}

template<typename M>
void DataSet<M>::getClassification(const tensor<float, M>& Y_predict) {
	tensor<float, M> result(Y.shape());
	tensor<float, host_memory_space> X_h = X;

	apply_binary_functor(result, Y, Y_predict, BF_SUBTRACT);

	ofstream cc("./Results/correctClassified4R.dat");
	ofstream mc("./Results/missClassified4R.dat");
	if(!cc || !mc){
		cerr<<"Cannot open the output file."<<endl;
		exit(1);
	}
	for (unsigned i = 0; i < result.shape(0); i++) {
		if (result[i] != 0.f) {
			for (int k = 0; k < this->n_dim; k++)
				mc<<X(i, k)<<" ";
			mc<<Y_predict[i]<<endl;
		} else {
			for (int k = 0; k < this->n_dim; k++)
				cc<<X(i, k)<<" ";
			cc<<Y[i]<<endl;
		}
	}
}
template<typename M>
void DataSet<M>::printData(char* fileName){
	ofstream fs(fileName);
	if(!fs){
		cerr<<"Cannot open the output file."<<endl;
		exit(1);
	}
	int portion = n_data/n_classes;
	tensor<float, dev_memory_space> tmp = X;
	for (int i = 0; i < this->n_classes; i++){
		for (int j = i*portion; j < portion*(i+1); j++) {
			for (int k = 0; k < this->n_dim; k++)
				fs<<X(j, k)<<" ";
			fs<<i<<endl;
		}
	}
}

template<typename M>
DataSet<M>::~DataSet() {
	// TODO Auto-generated destructor stub
}

template class DataSet<host_memory_space>;
template class DataSet<dev_memory_space>;
