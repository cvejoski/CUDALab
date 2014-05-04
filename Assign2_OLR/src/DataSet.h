/*
 * DataSet.h
 *
 *  Created on: May 2, 2014
 *      Author: cve
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <cuv.hpp>

using namespace cuv;

template<typename M>
class DataSet {
private:
	int n_dim;
	int n_data;
	int n_outputs;
	tensor<float, M> X;
	tensor<float, M> Y;
	tensor<float, M> w;
	tensor<float, M> b;
public:
	DataSet();
	DataSet(int, int, int, const tensor<float, M>&, const tensor<float, M>&);
	void createData();
	tensor<float, M> getData();
	tensor<float, M> getLabels();
	tensor<float, M> getW();
	tensor<float, M> getBias();
	virtual ~DataSet();
};

#endif /* DATASET_H_ */
