/*
 * OnLineLinearRegression.cpp
 *
 *  Created on: May 1, 2014
 *      Author: cve
 */

#include "OnLineLinearRegression.h"


template<typename M>
OnLineLinearRegression<M>::OnLineLinearRegression() {
	this->n_iter = 0;
	this->n_dim = 0;
	this->n_outputs = 0;
	this->l_rate = 0.0;

	this->b = NULL;
	this->w = NULL;
}

template<typename M>
OnLineLinearRegression<M>::OnLineLinearRegression(float l_rate, int n_iter,
		int n_outputs, int n_dim) {
	this->n_iter = n_iter;
	this->l_rate = l_rate;
	this->n_dim = n_dim;
	this->n_outputs = n_outputs;
	this->w = tensor<float, M>(extents[n_dim][n_outputs]);
	this->w = 0.f;
	this->b = tensor<float, M>(extents[1][n_outputs]);
	this->b = 0.f;
}

template<typename M>
void OnLineLinearRegression<M>::fit(const tensor<float, M>& X, const tensor<float, M>& y) {
	tensor<float, M> delta_w(w.shape());
	tensor<float, M> delta_b(b.shape());
	int c = 0;
	do {
		calcGradient(X, y, delta_w, delta_b);
		update_wb(delta_w, delta_b);
		//print loss function
		cout<<c<<" "<<norm2(delta_w)<<endl;
		c++;
	} while (!isConverging(0.0001, delta_w) && (c<=n_iter));
}

template<typename M>
tensor<float, M> OnLineLinearRegression<M>::predict(const tensor<float, M>& X_test) {
	tensor<float, M> result(extents[X_test.shape(0)][n_outputs]);
	prod(result, X_test, w, 'n', 'n', 1.f, 0.f);
	matrix_plus_row(result, b);
	return result;
}

template<typename M>
bool OnLineLinearRegression<M>::isConverging(const float& epsilon, const tensor<float, M>& delta_w) {
	if (norm2(delta_w) < epsilon)
		return true;
	else
		return false;
}

template<typename M>
void OnLineLinearRegression<M>::calcGradient(const tensor<float, M>& X, const tensor<float, M>& y,
		tensor<float, M>& delta_w, tensor<float, M>& delta_b) {
	delta_w = 0.f;
	delta_b = 0.f;
	tensor<float, M> tmpResult(extents[X.shape(0)][n_outputs]);
	prod(tmpResult, X, w, 'n', 'n', 1.f, 0.f);
	matrix_plus_row(tmpResult, b);
	apply_binary_functor(tmpResult, tmpResult, y, BF_SUBTRACT);
	prod(delta_w, X, tmpResult, 't', 'n', 2.f, 0.f);

	//calculating gradient descent for the bias
	tensor<float, M> tmpE(extents[X.shape(0)][1]);
	tmpE = 1.f;
	prod(tmpResult, X, w, 'n', 'n', 1.f, 0.f);
	matrix_plus_row(tmpResult, b);
	apply_binary_functor(tmpResult, tmpResult, y, BF_SUBTRACT);
	prod(delta_b, tmpE, tmpResult, 't', 'n', 2.f, 0.f);
}

template<typename M>
void OnLineLinearRegression<M>::update_wb(tensor<float, M> delta_w, tensor<float, M> delta_b) {
	apply_scalar_functor(delta_w, SF_MULT, l_rate);
	apply_binary_functor(w, w, delta_w, BF_SUBTRACT);
	apply_scalar_functor(delta_b, SF_MULT, l_rate);
	apply_binary_functor(b, b, delta_b, BF_SUBTRACT);
}

template<typename M>
void OnLineLinearRegression<M>::printLoss(const tensor<float, M>& delta_w, const int& step) {
//	ofstream fs("loss.dat");
//	if(!fs){
//		cerr<<"Cannot open the output file."<<endl;
//		exit(1);
//	}
//	tensor<M, host_memory_space> t = delta_w;
//	for (unsigned int i = 0; i<t.shape(0); i++)
//		fs<<step<<" "<<t[i]<<endl;
}

template<typename M>
tensor<float, M> OnLineLinearRegression<M>::getW(){
	return this->w;
}

template<typename M>
tensor<float, M> OnLineLinearRegression<M>::getB(){
	return this->b;
}

template<typename M>
OnLineLinearRegression<M>::~OnLineLinearRegression() {
	// TODO Auto-generated destructor stub
}

template class OnLineLinearRegression<host_memory_space>;
template class OnLineLinearRegression<dev_memory_space>;

