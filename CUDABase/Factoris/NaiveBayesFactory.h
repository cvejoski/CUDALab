/*
 * NaiveBayesFactory.h
 *
 *  Created on: Jun 2, 2014
 *      Author: cve
 */

#ifndef NAIVEBAYESFACTORY_H_
#define NAIVEBAYESFACTORY_H_

#include "../Algorithms/NaiveBayes.h"
#include "Factory.h"

template<typename M>
class NaiveBayesFactory : public Factory<M> {
public:
	MLalgorithm<M> * generate(float l_rate, float r_rate, int n_iter, int n_classes, int n_dim) {
		return new NaiveBayes<M>(n_classes, n_dim, r_rate);
	}
	virtual ~NaiveBayesFactory() {

	}
};


#endif /* NAIVEBAYESFACTORY_H_ */
