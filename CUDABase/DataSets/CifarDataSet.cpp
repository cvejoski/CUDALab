/*
 * CifarDataSet.cpp
 *
 *  Created on: Sep 18, 2014
 *      Author: cve
 */

#include "CifarDataSet.h"

template<typename M>
CifarDataSet<M>::CifarDataSet() :
DataSet<M>::DataSet() {
	this->X_test = NULL;
	this->Y_test = NULL;

	this->nTestData = 0;
	this->fine_labels = false;
}

template<typename M>
CifarDataSet<M>::CifarDataSet(const int& nDim, const int& nData, const int& nTestData, const int& nClasses, const bool& fine_labels)
: DataSet<M>::DataSet(nDim, nData, nClasses) {
	this->nTestData = nTestData;
	this->X_test = tensor<float, M>(extents[nTestData][nDim]);
	this->Y_test = tensor<float, M>(extents[nTestData][1]);
	this->fine_labels = fine_labels;
	createData();
}

template<typename M>
tensor<float, M> CifarDataSet<M>::getX_test() {
	return this->X_test;
}

template<typename M>
tensor<float, M> CifarDataSet<M>::getY_test() {
	return this->Y_test;
}

template<typename M>
void CifarDataSet<M>::createData() {
	std::string path = "/home/stud/cve/Documents/cifar-100-binary/";

	read(this->nData, "./cifar/train.bin", fine_labels, this->X, this->Y);
	read(nTestData, "./cifar/test.bin", fine_labels, this->X_test, this->Y_test);

}

template<typename M>
void CifarDataSet<M>::read(int itemCount, string path, const bool fine_labels, tensor<float, M>& X, tensor<float, M>& y) {
	ifstream dataStream;
	dataStream.open (path.c_str(), ios::binary );
	tensor<unsigned char, host_memory_space> data(extents[itemCount][3074]);

	dataStream.read((char*) data.ptr(), data.size());
	assert(dataStream.good());

		tensor_view<unsigned char, host_memory_space> coarseLabels_char_view =
				data[indices[index_range()][index_range(0, 1)]];
		tensor_view<unsigned char, host_memory_space> fineLabels_char_view =
				data[indices[index_range()][index_range(1, 2)]];
		tensor_view<unsigned char, host_memory_space> pixels_char_view =
				data[indices[index_range()][index_range(2, 3074)]];

		tensor<unsigned char, host_memory_space> coarseLabels_char =
				coarseLabels_char_view.copy();
		tensor<unsigned char, host_memory_space> fineLabels_char =
				fineLabels_char_view.copy();
		tensor<unsigned char, host_memory_space> pixels_char =
				pixels_char_view.copy();

		tensor<float, host_memory_space> coarseLabels(extents[itemCount][1]);
		tensor<float, host_memory_space> fineLabels(extents[itemCount][1]);
		tensor<float, host_memory_space> pixels(extents[itemCount][3072]);

		convert(coarseLabels, coarseLabels_char);
		convert(fineLabels, fineLabels_char);
		convert(pixels, pixels_char);

		X = pixels.copy();
		//cout<<fineLabels<<endl;
		//cout<<coarseLabels<<endl;
		y = fine_labels ? fineLabels.copy() : coarseLabels.copy();
		dataStream.close();
	}

template<typename M>
CifarDataSet<M>::~CifarDataSet() {

}

template class CifarDataSet<dev_memory_space>;
template class CifarDataSet<host_memory_space>;

