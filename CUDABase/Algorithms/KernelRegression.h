/*
 * KernelMethods.h
 *
 *  Created on: Jun 23, 2014
 *      Author: cve
 */

#ifndef KERNELREGRESSION_H_
#define KERNELREGRESSION_H_

#include "MLalgorithm.h"
#include "Kernels/Kernel.h"
#include <fstream>

template <typename M>
class KernelRegression : public MLalgorithm<M>{
private:
	int n_iter;
	int n_dim;
	int n_outputs;
	float l_rate;
	float r_rate;

	Kernel<M> * kernel;

	tensor<float, M> gramMatrix;
	tensor<float, M> alpha;

	tensor<float, M> X;

	tensor<float, M> b;

	void init();
	void calcGradient(const tensor<float, M>& X, const tensor<float, M>& y, tensor<float, M>& delta_w, tensor<float, M>& delta_b);
public:
	KernelRegression();

	/**
	* The constructor takes all parameters of the algorithm
	* @param l_classes
	* @param n_iter
	*/
	KernelRegression(float l_rate, float r_rate, int n_iter, int n_outputs, int n_dim, Kernel<M>* kernel);

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
	* The predict function gets only the test data and uses the
	* internal (=fitted) model to predict the outcome for the
	* test data and writes to file.
	*/
	void predict(const tensor<float, M>& X_test, char* fileName);

	/**
	 * Predict the data and return number of errors.
	 * @param X_test test data,
	 * @param Y_test labels for the test data,
	 * @return error from prediction.
	 */
	 double predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test);


	/**
	 * Print learned parameters on screen.
	 * Implementation depends on the type of the algorithm.
	 */
	void printParamToScreen();

	double loss(const tensor<float, M>& Y, const tensor<float, M>& Y_predicted);

	void plotParamAsImages();

	virtual ~KernelRegression();
};
#include "KernelRegression.cpp"
#endif /* KERNELREGRESSION_H_ */
