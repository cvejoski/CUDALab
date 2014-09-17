/*
 * Helper.h
 *
 *  Created on: Jul 7, 2014
 *      Author: cve
 */

#ifndef HELPER_H_
#define HELPER_H_

#include <cuv.hpp>
#include <vector>
#include <algorithm>

using namespace cuv;
using namespace std;

template<typename M>
class Helper {
public:
	Helper() {

	}

	void EuclidianDistance(tensor<float, M>& destination, const tensor<float, M>& A, const tensor<float, M>& B ) {
		tensor<float, M> tmp1(extents[A.shape(0)][1]);
		tensor<float, M> tmp2(extents[B.shape(0)][1]);
		tensor<float, M> e1(extents[B.shape(0)][1]);
		tensor<float, M> e2(extents[A.shape(0)][1]);

		e1 = 1.f;
		e2 = 1.f;

		prod(destination, A, B, 'n', 't', -2.f);
		reduce_to_col(tmp1, A, RF_ADD_SQUARED);
		prod(destination, tmp1, e1, 'n','t', 1.f, 1.f);
		reduce_to_col(tmp2, B, RF_ADD_SQUARED);
		prod(destination, e2, tmp2, 'n','t', 1.f, 1.f);
	}

	void addNoise(tensor<float, M>& destination, float deviation) {
//		initialize_mersenne_twister_seeds(0);
//		tensor<float, M> error(destination.shape());
//		error = 0.f;
//		add_rnd_normal(error);
//		error *= deviation;
//		destination += error;
		add_rnd_normal(destination, deviation);
	}

	void getSubGroup(tensor<float, M>& destination, const tensor<float, M>& source, const int& size) {
		unsigned int r = rand() % source.shape(0);
		if (r + size >= source.shape(0)) r -= size;
		tensor_view<float, M> slice = source[indices[index_range(r, r + size)]];
		destination = slice.copy();
	}

	void getRandomSubgroup(tensor<float, M>& result_X, tensor<float, M>& result_Y, const tensor<float, M>& source_X, const tensor<float, M> source_Y) {
		vector<int> data;
		for (unsigned i = 0; i < source_X.shape(0); i++) {
			data.push_back(i);
		}
		tensor<float, host_memory_space> h_result_X(result_X.shape());
		tensor<float, host_memory_space> h_result_Y(result_Y.shape());
		tensor<float, host_memory_space> h_source_X = source_X.copy();
		tensor<float, host_memory_space> h_source_Y = source_Y.copy();
		random_shuffle(data.begin(), data.end());
		for (unsigned int i = 0; i < result_X.shape(0); i++) {
			tensor_view<float, host_memory_space> slice_X = h_source_X[indices[index_range(data.at(i), data.at(i)+1)]];
			tensor_view<float, host_memory_space> slice_Y = h_source_Y[indices[index_range(data.at(i), data.at(i)+1)]];
			h_result_X[indices[index_range(i, i + 1)]] = slice_X;
			h_result_Y[indices[index_range(i, i + 1)]] = slice_Y;
		}
		result_X = h_result_X;
		result_Y = h_result_Y;
	}


	virtual ~Helper() {

	}
};

#endif /* HELPER_H_ */
