/*
 * LinearDataSet.cpp
 *
 *  Created on: Jun 23, 2014
 *      Author: cve
 */

#include "LinearDataSet.h"

template <typename M>
LinearDataSet<M>::LinearDataSet()
:DataSet<M>::DataSet() {
	this->w = NULL;
	this->b = NULL;
}

template <typename M>
LinearDataSet<M>::LinearDataSet(const int& nDim, const int& nData, const int& nClasses, const tensor<float, M>& w, const tensor<float, M>& b )
:DataSet<M>::DataSet(nDim, nData, nClasses) {
	this->w = w;
	this->b = b;
	createData();
}

template<typename M>
void LinearDataSet<M>::createData() {
	initialize_mersenne_twister_seeds(0);
	tensor<float, M> error(this->Y.shape());
	error = 0.f;
	add_rnd_normal(error);
	error *= .05f;
	fill_rnd_uniform(this->X);
	prod(this->Y, this->X, this->w, 'n', 'n', 1.f, 0.f);
	matrix_plus_row(this->Y, this->b);
	this->Y += error;
}

template<typename M>
tensor<float, M> LinearDataSet<M>::getW() {
	return w;
}

template<typename M>
tensor<float, M> LinearDataSet<M>::getBias() {
	return b;
}

template<typename M>
void LinearDataSet<M>::printToFile(char* fileName) {
		ofstream fs(fileName);
		if(!fs){
			cerr<<"Cannot open the output file."<<endl;
			exit(1);
		}
		tensor<float, host_memory_space> tmpX = this->X;
		tensor<float, host_memory_space> tmpY = this->Y;
		for (unsigned int i = 0; i < tmpX.shape(0); i++){
			for (unsigned int j = 0; j < tmpX.shape(1); j++) {
				fs<<tmpX(i, j)<<" ";
			}
			fs<<tmpY[i]<<endl;
		}
}

template <typename M>
LinearDataSet<M>::~LinearDataSet() {

}



