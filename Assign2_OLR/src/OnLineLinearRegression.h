/*
 * OnLineLinearRegression.h
 *
 *  Created on: May 1, 2014
 *      Author: cve
 */

#ifndef ONLINELINEARREGRESSION_H_
#define ONLINELINEARREGRESSION_H_

#include <sstream>
#include <fstream>
#include <iostream>
#include <cuv.hpp>
#include <stdlib.h>
#include "OnLineLinearRegression.h"

using namespace std;
using namespace cuv;

template<typename M>
class OnLineLinearRegression {
private:
	int n_iter;
	int n_dim;
	int n_outputs;
	float l_rate;

	tensor<float, M> b;
	tensor<float, M> w;

	bool isConverging(const float& epsilon, const tensor<float, M>& delta_w);
	void calcGradient(const tensor<float, M>& X, const tensor<float, M>& y, tensor<float, M>& delta_w, tensor<float, M>& delta_b);
	void update_wb(tensor<float, M> delata_w, const tensor<float, M> delta_b);
	void printLoss(const tensor<float, M>& delta_w, const int& ste);

public:
	OnLineLinearRegression();

	/**
	* The constructor takes all parameters of the algorithm
	* @param learningrate the size of the gradient steps
	* @param n_iter number of iterations to work for a given $X$
	*/
	OnLineLinearRegression(float l_rate, int n_iter, int n_outputs, int n_dim);

	/**
	* The fit function only takes the training data and the targets.
	* It builds (=fits) the model of the training data.
	*/
	void fit(const tensor<float, M>& X, const tensor<float, M>& y);
	/**
	* The predict function gets only the test data and uses the
	* internal (=fitted) model to predict the outcome for the
	* test data.
	*/
	tensor<float, M> predict(const tensor<float, M>& X_test);

	tensor<float, M> getB();

	tensor<float, M> getW();

	/**
	 * Only outputs in file for 2D case
	 */
	void saveLinesToFile(char* filename, const tensor<float, M>& w_0, const tensor<float, M>& b_0);

	virtual ~OnLineLinearRegression();
};



#endif /* ONLINELINEARREGRESSION_H_ */
