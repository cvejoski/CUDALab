/*
 * XORDataSet.h
 *
 *  Created on: Jul 8, 2014
 *      Author: cve
 */

#ifndef XORDATASET_H_
#define XORDATASET_H_

#include "DataSet.h"

template <typename M>
class XORDataSet : public DataSet<M> {
private:
	float sigma;
	bool multi;
	/**
	 * In this method should be implemented how we want to create the data.
	 */
	void createData();

public:
	/**
	 *Constructor for the base class.
	 *@param nDim provide the number of dimensions of the data,
	 *@param nData provide the number of instances that you want to create,
	 *@param covariance provide for the classes.
	 *@param mean provide for the classes.
	 */
	XORDataSet(const int& nDim, const int& nData, const float& sigma, const bool& multy);
	virtual ~XORDataSet();
};
#include "XORDataSet.cpp"
#endif /* XORDATASET_H_ */
