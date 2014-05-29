/*
 * LorRegFactory.h
 *
 *  Created on: May 22, 2014
 *      Author: cve
 */

#ifndef LORREGFACTORY_H_
#define LORREGFACTORY_H_

#include "../Algorithms/LogisticRegression.h"
#include "../Algorithms/LogisticRegression.cpp"
#include "Factory.h"

template<typename M>
class LogRegFactory : public Factory<M> {
public:
	MLalgorithm<M> * generate(float l_rate, float r_rate, int n_iter, int n_classes, int n_dim) {
		return new LogisticRegression<M>(l_rate, r_rate, n_iter, n_classes, n_dim);
	}
	virtual ~LogRegFactory() {

	}
};


#endif /* LORREGFACTORY_H_ */
