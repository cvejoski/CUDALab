/*
 * RegressionSigmoidDataSet.h
 *
 *  Created on: Jul 7, 2014
 *      Author: cve
 */

#ifndef REGRESSIONSIGMOIDDATASET_H_
#define REGRESSIONSIGMOIDDATASET_H_

#include "DataSet.h"
#include "../Algorithms/Helper.h"

template<typename M>
class RegressionSigmoidDataSet : public DataSet<M> {
private:
	float a, b;
	int subGroupSize;
	/**
	* In this method should be implemented how we want to create the data.
	*/
	void createData();
public:
	/**
	 *Constructor for the base class.
	 *@param nDim provide the number of dimensions of the data,
	 *@param nData provide the number of instances that you want to create,
	 *@param nClasses provide the number of classes you want to create.
	 *@param covariance provide for the classes.
	 *@param mean provide for the classes.
	 */
	RegressionSigmoidDataSet(const int& nDim, const int& nData, const int& nClasses, float a, float b, int subGroupSize);


	float get_a();

	float get_b();

	void printToFile(char* fileName);

	virtual ~RegressionSigmoidDataSet();
};
#include "RegressionSigmoidDataSet.cpp"
#endif /* REGRESSIONSIGMOIDDATASET_H_ */
