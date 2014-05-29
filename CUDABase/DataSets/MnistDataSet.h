/*
 * MnistDataSet.h
 *
 *  Created on: May 26, 2014
 *      Author: cve
 */

#ifndef MNISTDATASET_H_
#define MNISTDATASET_H_

#include "DataSet.h"

template<typename M>
class MnistDataSet : public DataSet<M> {
private:
	int nTestData;

	tensor<float, M> X_test;
	tensor<float, M> Y_test;
	/**
	 * In this method should be implemented how we want to create the data.
	 */
	void createData();
public:
	MnistDataSet();

	/**
	 *Constructor for the base class.
	 *@param nDim provide the number of dimensions of the data,
	 *@param nData provide the number of instances that you want to create,
	 *@param nClasses provide the number of classes you want to create.
	 *@param covariance provide for the classes.
	 *@param mean provide for the classes.
	 */
	MnistDataSet(const int& nDim, const int& nData, const int& nTestData, const int& nClasses);

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
	virtual ~MnistDataSet();
};
#include "MnistDataSet.cpp"
#endif /* MNISTDATASET_H_ */
