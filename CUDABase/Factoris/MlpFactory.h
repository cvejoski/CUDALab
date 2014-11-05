/*
 * MlpFactory.h
 *
 *  Created on: Sep 24, 2014
 *      Author: cve
 */

#ifndef MLPFACTORY_H_
#define MLPFACTORY_H_

#include "../Algorithms/MLPClassification.h"
#include "../Algorithms/Kernels/TanhKernel.h"
#include "Factory.h"

template<typename M>
class MlpFactory : public Factory<M> {
public:
	MLalgorithm<M> * generate(float l_rate, float r_rate, int n_iter, int n_classes, int n_dim ) {
				return NULL;
	}

	MLalgorithm<M> * generate(float l_rate, float r_rate, float sigma, int n_iter, int n_classes, int n_dim ) {
		return new MLPClassification<M>(l_rate, r_rate, n_iter, n_classes, n_dim, sigma, new TanhKernel<M>());
	}
	virtual ~MlpFactory() {

	}
};

#endif /* MLPFACTORY_H_ */
