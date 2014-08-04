/*
 * LinearKernel.h
 *
 *  Created on: Jul 8, 2014
 *      Author: cve
 */

#ifndef LINEARKERNEL_H_
#define LINEARKERNEL_H_

#include "Kernel.h"

template <typename M>
class LinearKernel : public Kernel<M> {
public:
	LinearKernel() {

	}

	void calculate(tensor<float, M>& result, const tensor<float, M>& X, const tensor<float, M>& O) {
		prod(result, X, O, 'n', 't', 1.f, 0.f);
	}

	virtual ~LinearKernel() {

	}
};

#endif /* LINEARKERNEL_H_ */
