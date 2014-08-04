/*
 * RegressionDataSetGaus.cpp
 *
 *  Created on: Jul 7, 2014LinearDataSet
 *      Author: cve
 */

#include "RegressionDataSetGaus.h"

template<typename M>
RegressionDataSetGaus<M>::RegressionDataSetGaus(const int& nDim, const int& nData, const int& nClasses, float sigma, int subGroupSize)
:DataSet<M>::DataSet(nDim, nData, nClasses) {
	this->sigma = sigma;
	this->subGroupSize = subGroupSize;
	createData();
}

template<typename M>
void RegressionDataSetGaus<M>::createData() {
	Helper<M> hlp;
	tensor<float, M> D(extents[this->X.shape(0)][subGroupSize]);
	tensor<float, M> alpha(extents[subGroupSize][this->nClasses]);
	tensor<float, M> m(extents[subGroupSize][this->X.shape(1)]);

	alpha = 0.f;
	fill_rnd_uniform(alpha);
	fill_rnd_uniform(this->X);

	hlp.getSubGroup(m, this->X, subGroupSize);
	hlp.EuclidianDistance(D, this->X, m);
	D /= -2.f * sigma * sigma;
	apply_scalar_functor(D, D, SF_EXP);

	tensor<float, M> Y_transposed(extents[this->nClasses][this->X.shape(0)]);
	prod(Y_transposed, alpha, D, 't', 't', 1.f, 0.f);
	transpose(this->Y, Y_transposed);

	add_rnd_normal(this->Y, 0.2);
}

template<typename M>
float RegressionDataSetGaus<M>::getSigma() {
	return this->sigma;
}

template<typename M>
void RegressionDataSetGaus<M>::printToFile(char* fileName) {
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
RegressionDataSetGaus<M>::~RegressionDataSetGaus() {
	// TODO Auto-generated destructor stub
}

template class RegressionDataSetGaus<host_memory_space>;
template class RegressionDataSetGaus<dev_memory_space>;

