/*
 * CifarDataSet.h
 *
 *  Created on: Sep 18, 2014
 *      Author: cve
 */

#ifndef CIFARDATASET_H_
#define CIFARDATASET_H_

#include "DataSet.h"

template<typename M>
class CifarDataSet : public DataSet<M> {
	int nTestData;
	bool fine_labels;

	tensor<float, M> X_test;
	tensor<float, M> Y_test;

	/**
	 * In this method should be implemented how we want to create the data.
	 */
	void createData();

	/**
	 * Converts the data to binary
	 */
	tensor<float, M> convertToBinary(const tensor<float, M>& X);


	void read(int itemCount, string path, const bool fine_labels, tensor<float, M>& X, tensor<float, M>& y);
public:
	CifarDataSet();
	/**
	 *Constructor for the base class.
	 *@param nDim provide the number of dimensions of the data,
	 *@param nData provide the number of instances that you want to create,
	 *@param nClasses provide the number of classes you want to create.
	 *@param covariance provide for the classes.
	 *@param mean provide for the classes.
	 */
	CifarDataSet(const int& nDim, const int& nData, const int& nTestData, const int& nClasses, const bool& fine_labels);

	/**
	 * Used for returning test data.
	 * @return tensor with test data.
	 */
	tensor<float, M> getX_test();

	/**
	 * Used for returning test labels.
	 * @return tensor with test labels.
	 */
	tensor<float, M> getY_test();

	/**
	 * Get data binary.
	 */
	tensor<float, M> getX_binary();

	/**
	* Get data binary.
	*/
	tensor<float, M> getX_test_binary();

	virtual ~CifarDataSet();
};

#endif /* CIFARDATASET_H_ */
