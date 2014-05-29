/*
 * NearesNeighbor.cpp
 *
 *  Created on: Apr 17, 2014
 *      Author: cve
 */

#include "NearestNeighbor.h"
#include <cuv.hpp>

#include <string>
using namespace cuv;

//namespace {
//	boost::shared_ptr<cuv::allocator> g_alloc(new cuv::pooled_cuda_allocator());
//	//boost::shared_ptr<cuv::allocator> g_alloc(new cuv::default_allocator());
//}

template<class T, class M>
NearestNeighbor<T, M>::NearestNeighbor(){
	this->numClasses = 0;
	this->numDim = 0;
	this->numTrainD = 0;
	this->numTestD = 0;
	this->covariance = NULL;
	this->mean = NULL;
	this->trainDS = NULL;
	this->trainL = NULL;
	this->testDS = NULL;
	this->testL = NULL;
	this->testResultLab = NULL;
}

template<class T, class M>
void NearestNeighbor<T, M>::initialize(short int numDim,
		short int numClasses, int numTrainD, int numTestD) {

	this->numDim = numDim;
	this->numClasses = numClasses;
	this->numTrainD = numTrainD;
	this->numTestD = numTestD;

	this->mean = new tensor<T, M>(extents[numClasses][numDim]);
	this->trainDS = new tensor<T, M>(extents[numClasses*numTrainD][numDim]);
	this->testDS = new tensor<T, M>(extents[numClasses*numTestD][numDim]);
	this->trainL = new tensor<T, M>(extents[numClasses*numTrainD]);
	this->testL = new tensor<T, M>(extents[numClasses*numTestD]);
	this->testResultLab = new tensor<T, M>(extents[numClasses*numTestD]);
	this->covariance = new tensor<T, M>(extents[numClasses][numDim][numDim]);

	*this->trainDS = 0.f;
	*this->testDS = 0.f;
	*this->trainL = 0.f;
	*this->testL = 0.f;
	*this->testResultLab = 0.f;
	*this->covariance = 0.f;

	(*mean)[0] = 1.f;
	(*mean)[1] = 0.7;
	(*mean)[2] = 1.3;
	(*mean)[3] = -1.0;
	(*mean)[4] = 3.1;
	(*mean)[5] =-1.7;
	(*mean)[6] = 2.9;
	(*mean)[7] = 1.f;

	(*covariance)(0, 0, 0) = 0.6;
	(*covariance)(0, 1, 1) = 0.6;
	(*covariance)(0, 1, 0) = 0.f;
	(*covariance)(0, 0, 1) = 0.f;
	(*covariance)(1, 0, 0) = 0.6;
	(*covariance)(1, 1, 1) = 0.6;
	(*covariance)(1, 1, 0) = 0.f;
	(*covariance)(1, 0, 1) = 0.f;
	(*covariance)(2, 0, 0) = 0.6;
	(*covariance)(2, 1, 1) = 0.6;
	(*covariance)(2, 1, 0) = 0.f;
	(*covariance)(2, 0, 1) = 0.f;
	(*covariance)(3, 0, 0) = 0.6;
	(*covariance)(3, 1, 1) = 0.6;
	(*covariance)(3, 1, 0) = 0.f;
	(*covariance)(3, 0, 1) = 0.f;
}

template<class T, class M>
void NearestNeighbor<T, M>::initializeMINST(short int numDim, int numTrainD, int numTestD) {

	this->numDim = numDim;
	this->numClasses = 1;
	this->numTrainD = numTrainD;
	this->numTestD = numTestD;

	this->trainDS = new tensor<T, M>(extents[numTrainD][numDim]);
	this->testDS = new tensor<T, M>(extents[numTestD][numDim]);
	this->trainL = new tensor<T, M>(extents[numTrainD]);
	this->testL =  new tensor<T, M>(extents[numTestD]);
	this->testResultLab =  new tensor<T, M>(extents[numTestD]);

	*this->trainDS = 0.f;
	*this->testDS = 0.f;
	*this->trainL = 0.f;
	*this->testL = 0.f;
	*this->testResultLab = 0.f;
}

template<class T, class M>
void NearestNeighbor<T, M>::createPoints(tensor<T, M>& data, tensor<T, M>& labels, int instances) {
	initialize_mersenne_twister_seeds(time(NULL));
	add_rnd_normal(data);

	for (short int i = 0; i<numClasses; i++) {
		tensor_view<T, M> t_vC = (*covariance)[indices[i]];
		tensor_view<T, M> t_vM = (*mean)[indices[i]];
		tensor_view<T, M> t_vD = data[indices[index_range(i*instances, (i+1)*instances)]];
		tensor_view<T, M> t_vL = labels[indices[index_range(i*instances, (i+1)*instances)]];
		t_vL = i;
		tensor<T, M> result(t_vD.shape());
		prod(result, t_vD, t_vC);
		matrix_plus_row(result, t_vM);
		data[indices[index_range(i*instances, (i+1)*instances)]] = result;
	}
}

template<class T, class M>
void NearestNeighbor<T, M>::createTrainDS() {
	createPoints(*trainDS, *trainL, numTrainD);
	saveToFile("Train.dat", *trainDS, *trainL);
}

template<class T, class M>
void NearestNeighbor<T, M>::createTestDS(){
	createPoints(*testDS, *testL, numTestD);
	saveToFile("Test.dat", *testDS, *testL);
}

template<typename T, typename M>
void NearestNeighbor<T, M>::calcDistance() {
	{
		tensor<T, M> D(extents[numClasses*numTrainD][numClasses*numTestD]);
		tensor<T, M> tmp1(extents[numClasses*numTrainD][1]);
		tensor<T, M> tmp2(extents[numClasses*numTestD][1]);
		tensor<T, M> e1(extents[numClasses*numTestD][1]);
		tensor<T, M> e2(extents[numClasses*numTrainD][1]);
		e1 = 1.f;
		e2 = 1.f;
		prod(D, *trainDS, *testDS, 'n', 't', -2.f);
		reduce_to_col(tmp1, *trainDS, RF_ADD_SQUARED);
		prod(D, tmp1, e1 , 'n','t', 1.f, 1.f);
		reduce_to_col(tmp2, *testDS, RF_ADD_SQUARED);
		prod(D, e2, tmp2, 'n','t', 1.f, 1.f);

		tensor<T, M> P(extents[1][numClasses*numTestD]);
		reduce_to_row(P, D, RF_ARGMIN);
		assignLabel(P, 0);
	}
	MisClassified(*testL, *testResultLab);
	saveCorrectIncorrect();
	saveToFile("TestClass.dat", *testDS, *testResultLab);
}

template<typename T, typename M>
void NearestNeighbor<T, M>::calcDistanceMINST(tensor<T, M>& test, int start) {

	tensor<T, M> D(extents[numTrainD][test.shape(0)]);
	tensor<T, M> tmp1(extents[numTrainD][1]);
	tensor<T, M> tmp2(extents[test.shape(0)][1]);
	tensor<T, M> e1(extents[test.shape(0)][1]);
	tensor<T, M> e2(extents[numTrainD][1]);
	e1 = 1.f;
	e2 = 1.f;
	prod(D, *trainDS, test, 'n', 't', -2.f);
	reduce_to_col(tmp1, *trainDS, RF_ADD_SQUARED);
	prod(D, tmp1, e1 , 'n','t', 1.f, 1.f);
	reduce_to_col(tmp2, test, RF_ADD_SQUARED);
	prod(D, e2, tmp2, 'n','t', 1.f, 1.f);

	tensor<T, M> P(extents[1][test.shape(0)]);
	reduce_to_row(P, D, RF_ARGMIN);

	assignLabel(P, start);
}

template<typename T, typename M>
void NearestNeighbor<T, M>::classifiedMINST() {

	tensor_view<T, M> tv1 = (*testDS)[indices[index_range(0, 3000)]];
	tensor<T, M> t1 = tv1;
	calcDistanceMINST(t1, 0);

	tensor_view<T, M> tv2 = (*testDS)[indices[index_range(3000, 6000)]];
	tensor<T, M> t2 = tv2.copy();
	calcDistanceMINST(t2, 3000);
	tensor_view<T, M> tv3 = (*testDS)[indices[index_range(6000, 9000)]];
	tensor<T, M> t3 = tv3.copy();
	calcDistanceMINST(t3, 6000);
	tensor_view<T, M> tv4 = (*testDS)[indices[index_range(9000, 10000)]];
	tensor<T, M> t4 = tv4.copy();
	calcDistanceMINST(t4, 9000);

	tensor<T, M> diff(extents[numTestD]);
	apply_binary_functor(diff, *testL, *testResultLab, BF_SUBTRACT);
	//MisClassified(*testL, *testResultLab);
}

template<typename T, typename M>
void NearestNeighbor<T, M>::assignLabel(tensor<T, M>& src, int start) {
	tensor<T, host_memory_space> t = src;
	for (unsigned int i = 0; i<src.size(); i++) {
		(*testResultLab)[i+start] = (*trainL)[*(t.ptr()+i)];
	}
}

template<typename T, typename M>
void NearestNeighbor<T, M>::saveToFile(char* fileName, tensor<T, M>& data, tensor<T, M>& labels) {
	ofstream fs(fileName);
	if(!fs){
		cerr<<"Cannot open the output file."<<endl;
		exit(1);
	}
	tensor<T, host_memory_space> t = data;
	tensor<T, host_memory_space> l = labels;
	for (unsigned int i = 0; i<t.shape(0); i++) {
		for (unsigned int j = 0; j<t.shape(1); j++)
			fs<<t(i, j)<< " ";
			fs<<l[i]<<endl;
		}
}

template<typename T, typename M>
void NearestNeighbor<T, M>::readMINST() {
	std::string path = "/home/local/datasets/MNIST";
	ifstream ftraind((path + "/train-images.idx3-ubyte").c_str());
	ifstream ftrainl((path + "/train-labels.idx1-ubyte").c_str());
	ifstream ftestd ((path + "/t10k-images.idx3-ubyte").c_str());
	ifstream ftestl ((path + "/t10k-labels.idx1-ubyte").c_str());

	char buf[16];
	ftraind.read(buf,16); ftrainl.read(buf, 8);
	ftestd.read(buf,16); ftestl.read(buf, 8);
	tensor<unsigned char, host_memory_space> traind(extents[numTrainD][numDim]);
	tensor<unsigned char, host_memory_space> trainl(extents[numTrainD]);
	tensor<unsigned char, host_memory_space> testd(extents[numTestD][numDim]);
	tensor<unsigned char, host_memory_space> testl(extents[numTestD]);
	ftraind.read((char*)traind.ptr(), traind.size());
	assert(ftraind.good());
	ftrainl.read((char*)trainl.ptr(), trainl.size());
	assert(ftrainl.good());
	ftestd.read((char*)testd.ptr(), testd.size());
	assert(ftestd.good());
	ftestl.read((char*)testl.ptr(), testl.size());
	assert(ftestl.good());
	convertD(traind, trainl, testd, testl);
}

template<typename T, typename M>
void NearestNeighbor<T, M>::convertD(tensor<unsigned char, host_memory_space>& traind, tensor<unsigned char,
		host_memory_space>& trainl, tensor<unsigned char, host_memory_space>& testd, tensor<unsigned char, host_memory_space>& testl) {

	tensor<float, host_memory_space> train_d(extents[numTrainD][numDim]);
	tensor<float, host_memory_space> train_l(extents[numTrainD]);
	tensor<float, host_memory_space> test_d(extents[numTestD][numDim]);
	tensor<float, host_memory_space> test_l(extents[numTestD]);
	// conversion to float:
	convert(train_d, traind);
	convert(train_l, trainl);
	convert(test_d, testd);
	convert(test_l, testl);
	*this->trainDS = train_d;
	*this->trainL = train_l;
	*this->testDS = test_d;
	*this->testL = test_l;
}
template<typename T, typename M>
int** NearestNeighbor<T, M>::MisClassified(tensor<T, M>& O, tensor<T, M>& C) {
	tensor<T, host_memory_space> original = O;
	tensor<T, host_memory_space> classified = C;
	//int result[309][2];
	int** result = new int*[330];
	for (int i = 0; i<309; i++)
		result[i] = new int[2];
	int miss = 0;

	cout<<"NUMBER OF RESULTS "<<classified.size();
	for (unsigned int i = 0; i<classified.size(); i++) {
		if (original[i] != classified[i]){
			cout<<"Element number: "<<i<<" Original class: "<<original[i]<<" Classified: "<<classified[i]<<endl;
			result[miss][0] = i;
			result[miss][1] = (int)classified[i];
			miss++;
		}
	}
	cout<<"Number of misclassification is "<<miss<<endl;
	return result;
}

template<typename T, typename M>
void NearestNeighbor<T, M>::saveCorrectIncorrect() {
	ofstream corr("TestCorrect.dat");
	ofstream incorr("TestIncorr.dat");
		if(!corr){
			cerr<<"Cannot open the output file TestCorrect.dat."<<endl;
			exit(1);
		}
		if(!corr){
					cerr<<"Cannot open the output file TestIncorr.dat."<<endl;
					exit(1);
				}
		tensor<T, host_memory_space> d = *testDS;
		tensor<T, host_memory_space> e = *testL;
		tensor<T, host_memory_space> r = *testResultLab;
		for (unsigned int i = 0; i<d.shape(0); i++) {
			if (e[i] == r[i]) {
				for (unsigned int j = 0; j<d.shape(1); j++)
					corr<<d(i, j)<< " ";
				corr<<e[i]<<endl;
			}
			else {
				for (unsigned int j = 0; j<d.shape(1); j++)
					incorr<<d(i, j)<< " ";
				incorr<<r[i]<<endl;
			}
		}
		corr.close();
		incorr.close();
}

template<typename T, typename M>
void NearestNeighbor<T, M>::exportMINST() {
	int** miss;
	miss = MisClassified(*testL, *testResultLab);
	for (int i = 0; i < 309; i++) {
		tensor<T, M> t = (*testDS)[indices[**(miss+i)]];
		int l = (*testL)[**(miss+i)];
		std::stringstream out;
		out<<"rec_"<<**(miss+i)<<"_";
		out << l;
		std::string a = out.str();
		l = *(*(miss+i)+1);
		out<<"_clas_";
		out<<l;
		std::string b = out.str();

		string filename = "./Results/Minst/Test_Record_" + b + ".ppm";
		const char* c = filename.c_str();
		ofstream f(c);
		f<<"P2\n28 28\n255\n";
		for (unsigned int i = 0; i<t.shape(0); i++) {
			f<<t[i]<<" ";
			if (i%28 == 0 && i != 0)
				f<<endl;
		}
		f.close();
	}
		//cout<<"Element number: "<<miss[i][0]<<" Classified: "<<miss[i][1]<<endl;

}

template<class T, class M>
NearestNeighbor<T, M>::~NearestNeighbor() {
	delete(testDS);
	delete(testL);
	delete(trainDS);
	delete(trainL);
	delete(testResultLab);
	delete(covariance);
	delete(mean);
}


