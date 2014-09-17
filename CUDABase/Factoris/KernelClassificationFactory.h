/*
 * KernelClassificationFactory.h
 *
 *  Created on: Sep 10, 2014
 *      Author: cve
 */

#ifndef KERNELCLASSIFICATIONFACTORY_H_
#define KERNELCLASSIFICATIONFACTORY_H_

#include "../Algorithms/KernelClassification.h"
#include "../Algorithms/Kernels/GaussianKernel.h"
#include "Factory.h"

template<typename M>
class KernelClassificationFactory : public Factory<M> {
private:

public:
	MLalgorithm<M> * generate(float l_rate, float r_rate, int n_iter, int n_classes, int n_dim ) {
			return NULL;
	}

	MLalgorithm<M> * generate(float l_rate, float r_rate, float sigma, int n_iter, int n_classes, int n_dim ) {
		return new KernelClassification<M>(l_rate, r_rate, n_iter, n_classes, n_dim, new GaussianKernel<M>(sigma));
	}

	virtual ~KernelClassificationFactory() {

	}
};

#endif /* KERNELCLASSIFICATIONFACTORY_H_ */
