/*
 * LinearDataSet.h
 *
 *  Created on: Jun 23, 2014
 *      Author: cve
 */

#ifndef LINEARDATASET_H_
#define LINEARDATASET_H_

#include "DataSet.h"

template<typename M>
class LinearDataSet : public DataSet<M> {
private:
	tensor<float, M> w;
	tensor<float, M> b;

	/**
	* In this method should be implemented how we want to create the data.
	*/
	void createData();
public:
	LinearDataSet();

	/**
	 *Constructor for the base class.
	 *@param nDim provide the number of dimensions of the data,
	 *@param nData provide the number of instances that you want to create,
	 *@param nClasses provide the number of classes you want to create.
	 *@param covariance provide for the classes.
	 *@param mean provide for the classes.
	 */
	LinearDataSet(const int& nDim, const int& nData, const int& nClasses, const tensor<float, M>& covariance, const tensor<float, M>& mean);

	/**
	 * Get weights
	 */
	tensor<float, M> getW();

	/**
	 * Get bias
	 */
	tensor<float, M> getBias();

	void printToFile(char* fileName);

	virtual ~LinearDataSet();
};
#include "LinearDataSet.cpp"
#endif /* LINEARDATASET_H_ */
