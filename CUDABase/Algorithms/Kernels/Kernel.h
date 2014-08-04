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
	virtual ~Kernel() {}
};

#endif /* KERNEL_H_ */
