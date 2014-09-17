/*
 * LogisticRegression.h
 *
 *  Created on: May 16, 2014
 *      Author: cve
 */

#ifndef LOGISTICREGRESSION_H_
#define LogisticRegression_H_

#include <cuv.hpp>
#include <iostream>

#include "../../CUDABase/Algorithms/MLalgorithm.h"

using namespace cuv;
using namespace std;

template<typename M>
class LogisticRegression : public MLalgorithm<M>{
private:
	int n_iter;
	int n_dim;
	int n_classes;
	float l_rate;
	float r_rate;

	tensor<float, M> b;
	tensor<float, M> w;

	/**
	 * We convert label matrix from 1 dim to 2 dim (n_data, n_classes)
	 * we set the classes to 1 for the data belonging to that class and all
	 * the other labels to 0
	 * @param Y all the labels for the data
	 */
	tensor<float, M> convertYtoM(const tensor<float, M>& Y);

	/**
	 * Calculates g(x) for all classes for all instances
	 * @param X all the instances
	 */
	tensor<float, M> calcLinearEq(const tensor<float, M>& X);

	/**
	 * Calculates sigma(g(x)) for the multiclass classification
	 * @param linearEq modifiable callculated all linear equations
	 *
	 */
	tensor<float, M> calcSigma(const tensor<float, M>& linearEq);

	/**
	 * Calculates sigma(g(x)) for the multiclass classification
	 * @param linearEq modifiable callculated all linear equations
	 *
	 */
	tensor<float, M> calcSigmaReduced(const tensor<float, M>& linearEq);

	/**
	 * Calculating gradient descent for multiclass
	 * @param X all the instances
	 * @param Y all the labels transformed for multiclass classification
	 */
	double calcGradientDesc_MC(const tensor<float, M>& X, const tensor<float, M>& Y);

	/**
	 * Calculating the error on every step
	 */
	double isConverging(const float& epsilon, const tensor<float, M>& delta_w);

	/**
	 * Calculate gradient descent for 2 class case
	 * @param X all the instances
	 * @param Y label for the instances
	 */
	double calcGradientDescent_2C(const tensor<float, M>& X, const tensor<float, M>& Y);

	/**
	 * Calculating the error on every step of gradient descent
	 */
	tensor<float, M> calcError(const tensor<float, M>& sigma, const tensor<float, M>& Y);

	/**
	 * Return tensor with predicted classes for 2 classes case
	 *
	 */
	tensor<float, M> predict_2C(const tensor<float, M>& X_test);

	/**
	 * Return tensor with predicted classes for multi-classes case
	 *
	 */
	tensor<float, M> predict_MC(const tensor<float, M>& X_test);

	/**
	 * Initializing weight parameters before learning
	 */
	void init();

	/**
	 * Number of mislcassified examples in training
	 */
	int missClassified(const tensor<float, M>& X, const tensor<float, M>& Y);

public:
	LogisticRegression();

	/**
	* The constructor takes all parameters of the algorithm
	* @param l_rate the size of the gradient steps
	* @param r_rate influence of regularization term
	* @param n_iter number of iterations to work for a given $X$
	*/
	LogisticRegression(float l_rate, float r_rate, int n_iter, int n_classes, int n_dim);

	/**
	* The fit function only takes the training data and the targets.
	* It builds (=fits) the model of the training data.
	*/
	void fit(const tensor<float, M>& X, const tensor<float, M>& Y);

	/**
	* The predict function gets only the test data and uses the
	* internal (=fitted) model to predict the outcome for the
	* test data.
	*/
	tensor<float, M> predict(const tensor<float, M>& X_test);

	/**
	* Returns the learned BIAS
	*
	*/
	tensor<float, M> getB();

	/**
	 * Returns the learned weights
	 */
	tensor<float, M> getW();

	/**
	 * Print learned parameters
	 */
	void printParamToScreen();

	/**
	 * Predict the data and return number of errors.
	 * @param X_test test data,
	 * @param Y_test labels for the test data,
	 * @return error from prediction.
	 */
	double predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test);

	virtual ~LogisticRegression();
};

#endif /* LogisticRegression_H_ */
