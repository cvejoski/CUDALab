/*
 * Kernel.h
 *
 *  Created on: Jun 23, 2014
 *      Author: cve
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#include <cuv.hpp>

using namespace cuv;

template <typename M>
class Kernel {
public:
	Kernel() {}
	virtual void calculate(tensor<float, M>& result, const tensor<float, M>& X, const tensor<float, M>& O) = 0;
	virtual void calculate(tensor<float, M>& result, const tensor<float, M>& X, const tensor<float, M>& O, const tensor<float, M>& B) = 0;
	virtual void calculate_derivative(tensor<float, M>& result, const tensor<float, M>& X) {};
	virtual ~Kernel() {}
};

#endif /* KERNEL_H_ */
