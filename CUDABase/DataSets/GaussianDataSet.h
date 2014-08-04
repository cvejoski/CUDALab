/*
 * GaussianDataSet.h
 *
 *  Created on: May 25, 2014
 *      Author: cve
 */

#ifndef GAUSSIANDATASET_H_
#define GAUSSIANDATASET_H_

#include "DataSet.h"

template <typename M>
class GaussianDataSet : public DataSet<M> {
private:
	tensor<float, M> mean;
	tensor<float, M> covariance;

	/**
	 * In this method should be implemented how we want to create the data.
	 */
	void createData();
public:
	GaussianDataSet();
	/**
	 *Constructor for the base class.
	 *@param nDim provide the number of dimensions of the data,
	 *@param nData provide the number of instances that you want to create,
	 *@param nClasses provide the number of classes you want to create.
	 *@param covariance provide for the classes.
	 *@param mean provide for the classes.
	 */
	GaussianDataSet(const int& nDim, const int& nData, const int& nClasses, const tensor<float, M>& covariance, const tensor<float, M>& mean);

	/**
	 *@return stored covariance.
	 */
	tensor<float, M> getCovariance();

	/**
	 *@return stored mean.
	 */
	tensor<float, M> getMean();

	virtual ~GaussianDataSet();
};

#include "./GaussianDataSet.cpp"
#endif /* GAUSSIANDATASET_H_ */

