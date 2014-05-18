/*
 * DataSet.h
 *
 *  Created on: May 16, 2014
 *      Author: cve
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <cuv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cuv;

template<typename M>
class DataSet {
	int n_dim;
		int n_data;
		int n_classes;

		tensor<float, M> mean;
		tensor<float, M> covariance;
		tensor<float, M> X;
		tensor<float, M> Y;

	public:
		DataSet();
		DataSet(int, int, int,  tensor<float, M>,  tensor<float, M>);
		void createData();
		tensor<float, M> getData();
		tensor<float, M> getLabels();
		tensor<float, M> getMean();
		tensor<float, M> getCovariance();
		void getClassification(const tensor<float, M>& Y_predict);
		void printData();
		void printData(char*);
		virtual ~DataSet();
};

#endif /* DATASET_H_ */
