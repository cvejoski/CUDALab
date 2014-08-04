/*
 * RegressionDataSetGaus.h
 *
 *  Created on: Jul 7, 2014
 *      Author: cve
 */

#ifndef REGRESSIONDATASETGAUS_H_
#define REGRESSIONDATASETGAUS_H_

#include "DataSet.h"
#include "../Algorithms/Helper.h"

template<typename M>
class RegressionDataSetGaus : public DataSet<M>{
private:
	float sigma;
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
	RegressionDataSetGaus(const int& nDim, const int& nData, const int& nClasses, float sigma, int subGroupSize);

	/**
	 * Get sigma
	 */
	float getSigma();

	void printToFile(char* fileName);

	virtual ~RegressionDataSetGaus();
};
#include "RegressionDataSetGaus.cpp"
#endif /* REGRESSIONDATASETGAUS_H_ */
