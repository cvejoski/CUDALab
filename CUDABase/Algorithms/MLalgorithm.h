/*
 * MLAlgorithm.h
 *
 *  Created on: May 19, 2014
 *      Author: cve
 */

#ifndef MLALGORITHM_H_
#define MLALGORITHM_H_

#include <cuv.hpp>
#include <iostream>

using namespace cuv;
using namespace std;

template<typename M>
class MLalgorithm {
public:
	/**
	* The fit function only takes the training data and the targets.
	* It builds (=fits) the model of the training data.
	*/
	virtual void fit(const tensor<float, M>& X, const tensor<float, M>& Y) = 0;

	/**
	* The predict function gets only the test data and uses the
	* internal (=fitted) model to predict the outcome for the
	* test data.
	*/
	virtual tensor<float, M> predict(const tensor<float, M>& X_test) = 0;

	/**
	 * Predict the data and return number of errors.
	 * @param X_test test data,
	 * @param Y_test labels for the test data,
	 * @return error from prediction.
	 */
	virtual double predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test) = 0;

	/**
	 * Print learned parameters on screen.
	 * Implementation depends on the type of the algorithm.
	 */
	virtual void printParamToScreen() = 0;

	virtual ~MLalgorithm() {

	}

};
#endif /* MLALGORITHM_H_ */
