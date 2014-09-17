/*
 * GaussianKernel.h
 *
 *  Created on: Jul 8, 2014
 *      Author: cve
 */

#ifndef GAUSSIANKERNEL_H_
#define GAUSSIANKERNEL_H_

#include "Kernel.h"
#include "../Helper.h"

template <typename M>
class GaussianKernel : public Kernel<M> {
private:
	float sigma;
public:
	GaussianKernel(const float& sigma) {
		this->sigma = sigma;
	}

	void calculate(tensor<float, M>& result, const tensor<float, M>& X, const tensor<float, M>& O) {
		Helper<M> hlp;
		hlp.EuclidianDistance(result, X, O);
		result /= -2.f * sigma * sigma;
		apply_scalar_functor(result, result, SF_EXP);
	}

	void calculate(tensor<float, M>& result, const tensor<float, M>& X, const tensor<float, M>& O, const tensor<float, M>& B) {


	}

	virtual ~GaussianKernel() {

	}
};

#endif /* GAUSSIANKERNEL_H_ */
