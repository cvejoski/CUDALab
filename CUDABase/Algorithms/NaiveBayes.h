/*
 * NaiveBayes.h
 *
 *  Created on: May 30, 2014
 *      Author: cve
 */

#ifndef NAIVEBAYES_H_
#define NAIVEBAYES_H_

#include "MLalgorithm.h"

#include <fstream>
#include <cstring>
#include <sstream>
#include <string>

template<typename M>
class NaiveBayes : public MLalgorithm<M> {
private:
	int n_dim;
	int n_classes;

	float floor;

	tensor<float, M> attributeProbabilities;
	tensor<float, M> classProbabilities;
	tensor<float, M> attributeProbabilitiesRever;


	/**
		 * Convert data labels to binary class labels
		 * @param n_data number of training data,
		 * @param Y training data labels.
		 * @return matrix #n_dataX#n_classes with binary classes
		 */
		tensor<float, M> convertToBinaryLabels(const tensor<float, M>& _Y);

	/**
	 * Calculate class probability.
	 * @param n_data number of training data,
	 * @param Y training data labels,
	 * @return vector with probabilities for every class.
	 */
	void calcClassesProbability(const unsigned& n_data, const tensor<float, M>& Y);

	tensor<unsigned char, M> getMaskForClass(const int& _class, const unsigned& n_data,	const tensor<float, M>& Y);

	tensor<float, M> getAttributeCount(const tensor<float, M>& X, const tensor<float, M>& Y);

	tensor<float, M> getClassCount(const unsigned & n_data, const tensor<float, M>& Y);

	void calcAttributeProbability(const tensor<float, M>& X, const tensor<float, M>& Y);

	void setFloor();

	/**
	* Number of mislcassified examples in training
	*/
	int missClassified(const tensor<float, M>& X, const tensor<float, M>& Y);

	tensor<unsigned, host_memory_space> vector_to_image_matrix(const tensor<float, host_memory_space>& vector);
	void save_as_images(const tensor<float, M>& data);

public:

	NaiveBayes();

	/**
	* The constructor takes all parameters of the algorithm
	* @param n_classes number of classes
	* @param n_dim number of dimensions
	*/
	NaiveBayes(int n_classes, int n_dim, const double& floor);

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

	void plotParamAsImages();

	virtual ~NaiveBayes();
};
#include "NaiveBayes.cpp"
#endif /* NAIVEBAYES_H_ */
