/*
 * KernelMethods.cpp
 *
 *  Created on: Jun 23, 2014
 *      Author: cve
 */

#include "KernelRegression.h"

template<typename M>
KernelRegression<M>::KernelRegression() {
	this->n_iter = 0;
	this->n_dim = 0;
	this->n_outputs = 0;
	this->l_rate = 0.0;
	this->r_rate = 0.f;
	this->gramMatrix = NULL;


	this->kernel = NULL;

}

template<typename M>
KernelRegression<M>::KernelRegression(float l_rate, float r_rate, int n_iter, int n_outputs, int n_dim, Kernel<M>* kernel) {
	this->n_iter = n_iter;
	this->l_rate = l_rate;
	this->n_dim = n_dim;
	this->n_outputs = n_outputs;
	this->r_rate = r_rate;

	this->kernel = kernel;

	this->b = tensor<float, M>(extents[1][n_outputs]);
	this->b = 0.f;
}

template<typename M>
void KernelRegression<M>::init() {
	initialize_mersenne_twister_seeds(time(NULL));
	this->alpha = 0.f;
	this->b = 0.f;
	add_rnd_normal(alpha);
	add_rnd_normal(b);

	alpha *= .01f;
	b *= .01f;
}

template<typename M>
void KernelRegression<M>::fit(const tensor<float, M>& X, const tensor<float, M>& Y){
	this->X = X;
	this->alpha = tensor<float, M>(extents[X.shape(0)][n_outputs]);
	this->alpha = 0.f;

	init();
	tensor<float, M> delta_alpha(this->alpha.shape());
	tensor<float, M> delta_b(this->b.shape());

	int c = 0;
	do {
		calcGradient(X, Y, delta_alpha, delta_b);
		//print loss function
		cout<<c<<" "<<loss(Y, predict(X))<<endl;
		c++;
	} while (c<=n_iter);
	//while (!isConverging(0.00001, delta_w) && (c<=n_iter));
}

template<typename M>
void KernelRegression<M>::calcGradient(const tensor<float, M>& X, const tensor<float, M>& Y,
		tensor<float, M>& delta_alpha, tensor<float, M>& delta_b) {
	delta_alpha = 0.f;
	delta_b = 0.f;

	tensor<float, M> tmpResult = predict(X);

	tmpResult -= Y;

	//calculating gradient descent for alpha
	prod(delta_alpha, gramMatrix, tmpResult, 'n', 'n', 2.f/X.shape(0), 0.f);

	//calculating gradient descent for the bias
 	reduce_to_row(delta_b, tmpResult, RF_ADD, 2.f/X.shape(0));

	//regularization
	delta_alpha += 2.f * this->alpha * this->r_rate;
	delta_b += 2.f * this->b * this->r_rate;

	//update alpha and b
	alpha -= delta_alpha * l_rate;
	b -= delta_b * l_rate;
}


template<typename M>
tensor<float, M> KernelRegression<M>::predict(const tensor<float, M>& X_test) {
	tensor<float, M> result(extents[X_test.shape(0)][n_outputs]);
	tensor<float, M> result_t(extents[n_outputs][X_test.shape(0)]);
	gramMatrix = tensor<float, M>(extents[X.shape(0)][X_test.shape(0)]);

	kernel->calculate(gramMatrix, X, X_test);
	prod(result_t, alpha, gramMatrix, 't', 'n', 1.f, 0.f);
	transpose(result, result_t);
	matrix_plus_row(result, b);

	return result;
}

template<typename M>
void KernelRegression<M>::predict(const tensor<float, M>& X_test, char* fileName) {
	tensor<float, M> Y = predict(X_test);
	ofstream fs(fileName);
		if(!fs){
			cerr<<"Cannot open the output file."<<endl;
			exit(1);
		}
		tensor<float, host_memory_space> tmpX = X_test;
		tensor<float, host_memory_space> tmpY = Y;
		for (unsigned int i = 0; i < tmpX.shape(0); i++){
			for (unsigned int j = 0; j < tmpX.shape(1); j++) {
				fs<<tmpX(i, j)<<" ";
			}
			fs<<tmpY[i]<<endl;
		}
}

template<typename M>
double KernelRegression<M>::predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test){
	return 0;
}

template<typename M>
void KernelRegression<M>::printParamToScreen() {

}

template<typename M>
double KernelRegression<M>::loss(const tensor<float, M>& Y, const tensor<float, M>& Y_predicted) {
	return this->averageLoss(Y, Y_predicted);
}

template<typename M>
KernelRegression<M>::~KernelRegression() {
	// TODO Auto-generated destructor stub
}

//template class KernelRegression<dev_memory_space>;
//template class KernelRegression<host_memory_space>;
