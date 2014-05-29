/*
 * DataSet.h
 *
 *  Created on: May 24, 2014
 *      Author: cve
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <cuv.hpp>
#include <fstream>

using namespace cuv;
using namespace std;

template<typename M>
class DataSet {

private:
	/**
	 * In this method should be implemented how we want to create the data.
	 */
	virtual void createData() = 0;

protected:
	int nDim;
	int nData;
	int nClasses;

	tensor<float, M> X;
	tensor<float, M> Y;

public:
	DataSet() {
		this->nDim = 0;
		this->nData = 0;
		this->nClasses = 0;

		this->X = NULL;
		this->Y = NULL;
	}

	/**
	 *Constructor for the base class.
	 *@param nDim provide the number of dimensions of the data,
	 *@param nData provide the number of instances that you want to create,
	 *@param nClasses provide the number of classes you want to create.
	 */
	DataSet(const int& nDim, const int& nData, const int& nClasses) {
		this->nDim = nDim;
		this->nData = nData;
		this->nClasses = nClasses;
	}

	/**
	 * Used for returning generated data.
	 * @return tensor with generated data.
	 */
	tensor<float, M> getData() {
		return this->X;
	}

	/**
	 * Used for returning generated labels.
	 * @return tensor with generated labels.
	 */
	virtual tensor<float, M> getLabels() {
		return this->Y;
	}

	/**
	 * Output the data and labels to terminal.
	 */
	void printToScreen() {
		tensor<float, dev_memory_space> tmp = X;
		int portion = nData/nClasses;
		for (int i = 0; i < this->nClasses; i++){
			cout<<"CLASS "<<i<<":\n";
			for (int j = i*portion; j < portion*(i+1); j++) {
				for (int k = 0; k < this->nDim; k++)
					cout<<X(j, k)<<" ";
				cout<<endl;
			}
		}
	}

	/**
	 * Output the data and labels to file.
	 * @param fileName location and name of the file
	 * in witch you want the data to be exported.
	 */
	void printToFile(char* fileName) {
		ofstream fs(fileName);
		if(!fs){
			cerr<<"Cannot open the output file."<<endl;
			exit(1);
		}
		int portion = nData/nClasses;
		tensor<float, dev_memory_space> tmp = X;
		for (int i = 0; i < this->nClasses; i++){
			for (int j = i*portion; j < portion*(i+1); j++) {
				for (int k = 0; k < this->nDim; k++)
					fs<<X(j, k)<<" ";
				fs<<i<<endl;
			}
		}
	}

	virtual ~DataSet() {

	}
};


#endif /* DATASET_H_ */
