/*
 * RegressionSigmoidDataSet.cpp
 *
 *  Created on: Jul 7, 2014
 *      Author: cve
 */

#include "RegressionSigmoidDataSet.h"

template<typename M>
RegressionSigmoidDataSet<M>::RegressionSigmoidDataSet(const int& nDim, const int& nData, const int& nClasses, float a, float b, int subGroupSize)
:DataSet<M>::DataSet(nDim, nData, nClasses) {
	this->a = a;
	this->b = b;
	this->subGroupSize = subGroupSize;
	createData();
}

template<typename M>
void RegressionSigmoidDataSet<M>::createData() {
	Helper<M> hlp;
	tensor<float, M> D(extents[this->X.shape(0)][subGroupSize]);
	tensor<float, M> alpha(extents[subGroupSize][this->nClasses]);
	tensor<float, M> m(extents[subGroupSize][this->X.shape(1)]);

	alpha = 0.f;
	fill_rnd_uniform(alpha);
	this->X = 0;

	add_rnd_normal(this->X);

	hlp.getSubGroup(m, this->X, subGroupSize);
	prod(D, this->X, m, 'n', 't', 1.f, 0.f);
	D *= this->a;
	D += this->b;
	apply_scalar_functor(D, D, SF_TANH);

	tensor<float, M> Y_transposed(extents[this->nClasses][this->X.shape(0)]);
	prod(Y_transposed, alpha, D, 't', 't', 1.f, 0.f);
	transpose(this->Y, Y_transposed);

	add_rnd_normal(this->Y, 0.2);
}

template<typename M>
float RegressionSigmoidDataSet<M>::get_a() {
	return this->a;
}

template<typename M>
float RegressionSigmoidDataSet<M>::get_b() {
	return this->b;
}

template<typename M>
void RegressionSigmoidDataSet<M>::printToFile(char* fileName) {
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

template<typename M>
RegressionSigmoidDataSet<M>::~RegressionSigmoidDataSet() {

}
template class RegressionSigmoidDataSet<host_memory_space>;
template class RegressionSigmoidDataSet<dev_memory_space>;
