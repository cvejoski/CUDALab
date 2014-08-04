/*
 * XORDataSet.cpp
 *
 *  Created on: Jul 8, 2014
 *      Author: cve
 */

#include "XORDataSet.h"

template <typename M>
XORDataSet<M>::XORDataSet(const int& nDim, const int& nData, const float& sigma, const bool& multi) :
DataSet<M>::DataSet(nDim, nData, 4) {
	this->sigma = sigma;
	this->multi = multi;
	createData();
}

template<typename M>
void XORDataSet<M>::createData() {
	this->X = 0.f;
	int portion = this->nData / this->nClasses;
	add_rnd_normal(this->X);

	tensor<float, M> mean1(extents[4][2]);


	mean1[0] = -2.f;
	mean1[1] = 2.f;
	mean1[2] = -2.f;
	mean1[3] = -2.f;
	mean1[4] = 1.5;
	mean1[5] = -2.f;
	mean1[6] = 1.f;
	mean1[7] = 1.f;

	for (short int i = 0; i < this->nClasses; i++) {
		tensor_view<float, M> t_vM = mean1[indices[i]];
		tensor_view<float, M> t_vX = this->X[indices[index_range(portion*i, (i+1)*portion)]];
		tensor_view<float, M> t_vY = this->Y[indices[index_range(portion*i, (i+1)*portion)]];
		if (i == 0 || i == 2 ) {
			t_vY = 0;
		} else if (this->multi && i == 3) {
			t_vY = 2;
		} else {
			t_vY = 1;
		}
		tensor<float, M> result(t_vX.shape());
		result = t_vX.copy();

		matrix_plus_row(result, t_vM);
		this->X[indices[index_range(portion*i, (i+1)*portion)]] = result;
		this->X *= sigma;
	}
}

template <typename M>
XORDataSet<M>::~XORDataSet() {

}

template class XORDataSet<dev_memory_space>;
template class XORDataSet<host_memory_space>;
