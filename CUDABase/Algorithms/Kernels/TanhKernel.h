/*
 * TanhKernel.h
 *
 *  Created on: Aug 6, 2014
 *      Author: cve
 */

#ifndef TANHKERNEL_H_
#define TANHKERNEL_H_

#include "Kernel.h"

template <typename M>
class TanhKernel : public Kernel<M> {

public:
	TanhKernel() {

	}

	void calculate(tensor<float, M>& result, const tensor<float, M>& X, const tensor<float, M>& O) {

	}

	void calculate(tensor<float, M>& result, const tensor<float, M>& X, const tensor<float, M>& O, const tensor<float, M>& B) {
		prod(result, X, O, 'n', 'n', 1.f, 0.f);
		matrix_plus_row(result, B);
		apply_scalar_functor(result, result, SF_TANH);
	}

	void calculate_derivative(tensor<float, M>& result, const tensor<float, M>& X) {
		result = 1.f - X * X;
	}

	virtual ~TanhKernel() {

	}
};



#endif /* TANHKERNEL_H_ */
