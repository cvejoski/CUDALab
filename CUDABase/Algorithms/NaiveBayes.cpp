/*
 * NaiveBayes.cpp
 *
 *  Created on: May 30, 2014
 *      Author: cve
 */

#include "NaiveBayes.h"

template<typename M>
NaiveBayes<M>::NaiveBayes() {
	this->n_classes = 0;
	this->n_dim = 0;
	this->floor = 0.f;
	this->attributeProbabilities = NULL;
	this->classProbabilities = NULL;

	this->attributeProbabilitiesRever = NULL;
}

template<typename M>
NaiveBayes<M>::NaiveBayes(int n_classes, int n_dim, const double& floor) {
	this->n_classes = n_classes;
	this->n_dim = n_dim;
	this->floor = floor;

	this->attributeProbabilities = tensor<float, M>(extents[n_classes][n_dim]);
	this->classProbabilities = tensor<float, M>(extents[n_classes]);
	this->attributeProbabilitiesRever = tensor<float, M>(extents[n_classes][n_dim]);
}

template<typename M>
tensor<float, M> NaiveBayes<M>::convertToBinaryLabels(const tensor<float, M>& _Y) {
	tensor<float, M> result(extents[_Y.shape(0)][n_classes]);
	tensor<float, M> tmp(extents[_Y.shape(0)]);
	for (int i = 0; i < n_classes; i++) {
		apply_scalar_functor(tmp, _Y, SF_EQ, (float) i);
		result[indices[index_range()][index_range(i, i + 1)]] = tmp;
	}
	return result;
}

template<typename M>
tensor<float, M> NaiveBayes<M>::getClassCount(const unsigned & n_data, const tensor<float, M>& Y) {
	tensor<float, M> result(extents[n_classes]);
	reduce_to_row(result, Y);
	return result;
}

template<typename M>
void NaiveBayes<M>::calcClassesProbability(const unsigned& n_data, const tensor<float, M>& Y) {
	this->classProbabilities = getClassCount(n_data, Y) / (float) (n_data);
	apply_scalar_functor(classProbabilities, classProbabilities, SF_LOG);
}

template<typename M>
tensor<unsigned char, M> NaiveBayes<M>::getMaskForClass(const int& _class, const unsigned& n_data, const tensor<float, M>& Y) {
	tensor<unsigned char, M> result(extents[n_data][n_dim]);
	tensor<float, M> mask_float(extents[n_data][n_dim]);
	mask_float = 0.f;
	tensor<float, M> tmp = Y[indices[index_range()][index_range(_class, _class + 1)]].copy();
	matrix_plus_col(mask_float, tmp);
	apply_scalar_functor(result, mask_float, SF_GT, 0.f);
	return result;
}

template<typename M>
tensor<float, M> NaiveBayes<M>::getAttributeCount(const tensor<float, M>& X, const tensor<float, M>& Y) {
	tensor<float, M> result(extents[n_classes][n_dim]);
	tensor<unsigned char, M> mask;
	result = 0.f;
	for (int i = 0; i < n_classes; i++) {
		mask = getMaskForClass(i, X.shape(0), Y);
		{
			tensor<float, M> tmp(extents[X.shape(0)][n_dim]);
			tensor<float, M> tmp_row(extents[n_dim]);
			tmp = 0.f;
			tmp_row = 0.f;
			apply_scalar_functor(tmp, X, SF_EQ, 1.f, &mask);
			reduce_to_row(tmp_row, tmp);
			result[indices[index_range(i, i + 1)][index_range()]] =	tmp_row;

		}
	}
	return result;
}

template<typename M>
void NaiveBayes<M>::calcAttributeProbability(const tensor<float, M>& X, const tensor<float, M>& Y) {
	attributeProbabilities = getAttributeCount(X, Y);
	matrix_divide_col(attributeProbabilities, getClassCount(X.shape(0), Y));
	setFloor();
	attributeProbabilitiesRever = 1.f - attributeProbabilities;
	apply_scalar_functor(attributeProbabilities, attributeProbabilities, SF_LOG);
	apply_scalar_functor(attributeProbabilitiesRever, attributeProbabilitiesRever, SF_LOG);
}

template<typename M>
void NaiveBayes<M>::fit(const tensor<float, M>& X, const tensor<float, M>& Y) {
	tensor<float, M> Y_binary = convertToBinaryLabels(Y);
	calcClassesProbability(X.shape(0), Y_binary);
	calcAttributeProbability(X, Y_binary);
	cout<<missClassified(X, Y)<<endl;
}

template<typename M>
void NaiveBayes<M>::setFloor() {
	apply_scalar_functor(attributeProbabilities, attributeProbabilities, SF_ADD, floor);
	attributeProbabilities /= (1.f + floor);
}

template<typename M>
int NaiveBayes<M>::missClassified(const tensor<float, M>& X, const tensor<float, M>& Y) {
	int result = 0;
	tensor<float, M> diff(Y.shape());
	apply_binary_functor(diff, Y, predict(X), BF_EQ);
	result = Y.shape(0) - sum(diff);
	return result;
}

template<typename M>
tensor<float, M> NaiveBayes<M>::predict(const tensor<float, M>& X_test) {
	tensor<float, M> result(extents[X_test.shape(0)]);
	tensor<float, M> tmp(extents[X_test.shape(0)][n_classes]);

	prod(tmp, X_test, attributeProbabilities, 'n', 't');
	prod(tmp, 1.f - X_test, attributeProbabilitiesRever, 'n', 't', 1.f, 1.f);


	matrix_plus_row(tmp, -1.f * classProbabilities);


	reduce_to_col(result, tmp, RF_ARGMAX);

	return result;
}

template<typename M>
double NaiveBayes<M>::predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test) {
	return missClassified(X_test, Y_test);
}

template<typename M>
void NaiveBayes<M>::printParamToScreen() {

}

template<typename M>
void NaiveBayes<M>::plotParamAsImages() {
	apply_scalar_functor(attributeProbabilities, SF_EXP);
	attributeProbabilities *= 255.f;
	save_as_images(attributeProbabilities);
}

template <typename M>
void NaiveBayes<M>::save_as_images(const tensor<float, M>& data) {
	tensor<float, host_memory_space> data_host = data;
	tensor<float, host_memory_space> row(extents[data.shape(0)]);
	stringstream ss;

	for (unsigned i = 0, ii = data.shape(0); i < ii; ++i) {
		ss << i;
		row = data_host[indices[i]];


		tensor<unsigned, host_memory_space> image = vector_to_image_matrix(row);

		string filename = "./Results/Bayes/Test_Record_" + ss.str() + ".ppm";


		const char* c = filename.c_str();
		ofstream f(c);
		f<<"P2\n28 28\n255\n";
		for (unsigned int i = 0; i<image.shape(0); i++) {
			for (unsigned int j = 0; j<image.shape(1); j++) {
			f<<image(i,j)<<" ";
			}
			f<<endl;
		}
		f.close();

		ss.str("");
	}

	std::cout << "Images saved.\n";
}

template<typename M>
tensor<unsigned, host_memory_space> NaiveBayes<M>::vector_to_image_matrix(const tensor<float, host_memory_space>& vector) {
	const unsigned resolution = 28;
	tensor<unsigned, host_memory_space> image(extents[resolution][resolution]);

	for (unsigned i = 0; i < resolution; ++i) {
		for (unsigned j = 0; j < resolution; ++j) {
			image(i, j) = vector[i * resolution + j];
		}
	}

	return image;
}


template<typename M>
NaiveBayes<M>::~NaiveBayes() {
	// TODO Auto-generated destructor stub
}

template class NaiveBayes<host_memory_space>;
template class NaiveBayes<dev_memory_space>;

