/*
 * SigmoidKernel.h
 *
 *  Created on: Jul 8, 2014
 *      Author: cve
 */

#ifndef SIGMOIDKERNEL_H_
#define SIGMOIDKERNEL_H_

#include "Kernel.h"

template <typename M>
class SigmoidKernel : public Kernel<M> {
private:
	float a, b;
public:
	SigmoidKernel(const float& a, const float& b) {
		this->a = a;
		this->b = b;
	}

	void calculate(tensor<float, M>& result, const tensor<float, M>& X, const tensor<float, M>& O) {
		prod(result, X, O, 'n', 't', 1.f, 0.f);
		result *= this->a;
		result += this->b;
		apply_scalar_functor(result, result, SF_TANH);
	}

	void calculate(tensor<float, M>& result, const tensor<float, M>& X, const tensor<float, M>& O, const tensor<float, M>& B) {

	}

	virtual ~SigmoidKernel() {

	}
};

#endif /* SIGMOIDKERNEL_H_ */
