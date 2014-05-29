/*
 * GaussianDataSet.cpp
 *
 *  Created on: May 25, 2014
 *      Author: cve
 */

#include "GaussianDataSet.h"

template<typename M>
GaussianDataSet<M>::GaussianDataSet() :
DataSet<M>::DataSet() {
	this->covariance = NULL;
	this->mean = NULL;
}

template<typename M>
GaussianDataSet<M>::GaussianDataSet(const int& nDim, const int& nData, const int& nClasses, const tensor<float, M>& covariance, const tensor<float, M>& mean) :
DataSet<M>::DataSet(nDim, nData, nClasses) {
	this->covariance = covariance;
	this->mean = mean;
	this->X = tensor<float, M>(extents[nData][nDim]);
	this->Y = tensor<float, M>(extents[nData][1]);
	createData();
}

template<typename M>
tensor<float, M> GaussianDataSet<M>::getCovariance() {
	return this->covariance;
}

template<typename M>
tensor<float, M> GaussianDataSet<M>::getMean() {
	return this->mean;
}

template<typename M>
void GaussianDataSet<M>::createData() {
	initialize_mersenne_twister_seeds(0);
	this->X = 0.f;
	add_rnd_normal(this->X);
	int portion = this->nData/this->nClasses;
	for (short int i = 0; i < this->nClasses; i++) {
		tensor_view<float, M> t_vC = covariance[indices[i]];
		tensor_view<float, M> t_vM = mean[indices[i]];
		tensor_view<float, M> t_vX = this->X[indices[index_range(portion*i, (i+1)*portion)]];
		tensor_view<float, M> t_vY = this->Y[indices[index_range(portion*i, (i+1)*portion)]];
		t_vY = i;
		tensor<float, M> result(t_vX.shape());
		result = t_vX.copy();
		//prod(result, t_vX, t_vC);
		matrix_plus_row(result, t_vM);
		this->X[indices[index_range(portion*i, (i+1)*portion)]] = result;
	}
}

template<typename M>
GaussianDataSet<M>::~GaussianDataSet() {

}
