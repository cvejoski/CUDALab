/*
 * CrossValidator.h
 *
 *  Created on: May 20, 2014
 *      Author: cve
 */

#ifndef CROSSVALIDATOR_H_
#define CROSSVALIDATOR_H_

#include <cuv.hpp>
#include <iostream>
#include <vector>
#include "../Factoris/Factory.h"

using namespace std;
using namespace cuv;

template<typename M>
class CrossValidator {
private:
	unsigned k_fold;
	unsigned nClasses;
	unsigned nIterations;

	vector<float> l_rates;
	vector<float> r_rates;
	vector<float> sigma;

	MLalgorithm<M> * bestModel;
	Factory<M> * factory;

	/**
	 * Splits the data in k parts.
	 * @param X data to be split.
	 * @return vector of k tensor_view.
	 */
	vector<tensor_view<float, M> > kSplit(const tensor<float, M>& X);

	/**
	 * Splits the data in k parts.
	 * @param X data to be split.
	 * @return vector of k tensor_view.
	 */
	vector<tensor<float, M> > kSplitAlternative(const tensor<float, M>& X, const unsigned int& i);

	/**
	 * Fit current model with given training data and labels
	 * @param X training data
	 * @param Y labels
	 * @param model is the instance of the algorithm for the current model
	 * @return average error of k folds
	 */
	float fitModel(const tensor<float, M>& X, const tensor<float, M>& Y, MLalgorithm<M>* model);

	/**
	 * Creates a model for every hyper parameter.
	 * @param X test data,
	 * @param meanError stores the average error from the k fold,
	 * @param Y labels,
	 * @param models store learned models.
	 */
	void fitHyperParamethers(const tensor<float, M>& X, vector<float>& meanError,
			const tensor<float, M>& Y, vector<MLalgorithm<M> *>& models);

	/**
	 * Creates a model for every hyper parameter.
	 * @param X test data,
	 * @param meanError stores the average error from the k fold,
	 * @param Y labels,
	 * @param models store learned models.
	 */
	void fitHyperParamethersSigma(const tensor<float, M>& X, vector<float>& meanError,
			const tensor<float, M>& Y, vector<MLalgorithm<M> *>& models);

	/**
	 * Find best model among all models.
	 * @param meanError for all models,
	 * @param models all learned models,
	 * @return model with smallest average error.
	 */
	MLalgorithm<M>* findBestModel(const vector<float>& meanError, const vector<MLalgorithm<M>* >& models);

public:
	CrossValidator();

	/**
	* The constructor takes all parameters of the algorithm
	* @param l_rate the size of the gradient steps
	* @param r_rate influence of regularization term
	* @param k_fold number of k splits for the data
	* @param factory for generating instances of desired algorithm
	*/
	CrossValidator(const vector<float>& l_rates, const vector<float>& r_rates, const unsigned& nClasses, const unsigned& nIterations, const unsigned& k_fold, Factory<M> * factory);

	/**
	* The constructor takes all parameters of the algorithm
	* @param l_rate the size of the gradient steps
	* @param r_rate influence of regularization term
	* @param sigma for Gaussian Kernel
	* @param k_fold number of k splits for the data
	* @param factory for generating instances of desired algorithm
	*/
	CrossValidator(const vector<float>& l_rates, const vector<float>& r_rates, const vector<float>& sigma, const unsigned& nClasses, const unsigned& nIterations, const unsigned& k_fold, Factory<M> * factory);

	/**
	* The fit function only takes the training data and the targets.
	* It builds (=fits) the model of the training data.
	*/
	void fit(const tensor<float, M>& X, const tensor<float, M>& Y);

	/**
	* The fit function only takes the training data and the targets.
	* It builds (=fits) the model of the training data.
	*/
	void fit_kernel(const tensor<float, M>& X, const tensor<float, M>& Y);

	/**
	* The predict function gets only the test data and uses the
	* internal (=fitted) model to predict the outcome for the
	* test data.
	*/
	tensor<float, M> predict(const tensor<float, M>& X_test);

	/**
	 * Predict the data and return number of errors.
	 * @param X_test test data,
	 * @param Y_test labels for the test data,
	 * @return error from prediction.
	 */
	double predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test);

	/**
	 * Get best model
	 */
	void printBestModel();
	virtual ~CrossValidator();
};
//#include "CrossValidator.cpp"
#endif /* CROSSVALIDATOR_H_ */
