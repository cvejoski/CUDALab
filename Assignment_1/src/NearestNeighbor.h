/*
 * NearesNeighbor.h
 *
 *  Created on: Apr 17, 2014
 *      Author: cve
 */

#ifndef NEARESNEIGHBOR_H_
#define NEARESNEIGHBOR_H_

#include <map>
#include <cuv.hpp>

#include <string>


using namespace cuv;
using namespace std;

template <class T, class M>
class NearestNeighbor {
private:
	short int numDim, numClasses;
	int numTrainD, numTestD;

	tensor<T, M> *trainDS;
	tensor<T, M> *trainL;
	tensor<T, M> *testDS;
	tensor<T, M> *testL;
	tensor<T, M> *testResultLab;
	tensor<T, M> *mean;
	tensor<T, M> *covariance;

	void createPoints(tensor<T, M>&, tensor<T, M>&, int);
	void assignLabel(tensor<T, M>&, int start);
	void convertD(tensor<unsigned char, host_memory_space>&, tensor<unsigned char, host_memory_space>&,
			tensor<unsigned char, host_memory_space>&, tensor<unsigned char, host_memory_space>&);
	int** MisClassified(tensor<T, M>&, tensor<T, M>&);
	void saveCorrectIncorrect();

public:
	NearestNeighbor();
	void initialize(short int numDim, short int numClasses, int numTrainD, int numTestD);
	void initializeMINST(short int numDim, int numTrainD, int numTestD);
	void createTrainDS();
	void createTestDS();
	void calcDistance();
	void calcDistanceMINST(tensor<T, M>&, int start);
	void classifiedMINST();
	void exportMINST();
	void saveToFile(char*, tensor<T, M>&, tensor<T, M>&);
	void readMINST();
	~NearestNeighbor();
};

#endif /* NEARESNEIGHBOR_H_ */
