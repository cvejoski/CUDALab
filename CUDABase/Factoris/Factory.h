/*
 * Factory.h
 *
 *  Created on: May 22, 2014
 *      Author: cve
 */

#ifndef FACTORY_H_
#define FACTORY_H_

#include "../Algorithms/MLalgorithm.h"

template<typename M>
class Factory {
public:
	virtual MLalgorithm<M> * generate(float l_rate, float r_rate, int n_iter, int n_classes, int n_dim) = 0;
	virtual MLalgorithm<M> * generate(float l_rate, float r_rate, float sigma, int n_iter, int n_classes, int n_dim) {
		return NULL;
	}
	virtual ~Factory() {

	}
};

#endif /* FACTORY_H_ */
