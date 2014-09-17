/*
 * MLPClassification.cpp
 *
 *  Created on: Aug 6, 2014
 *      Author: cve
 */

#include "MLPClassification.h"

template<typename M>
MLPClassification<M>::MLPClassification() {
	this->n_iter = 0;
	this->n_dim = 0;
	this->n_classes = 0;
	this->n_hidden_units = 0;

	this->l_rate = 0.0;
	this->r_rate = 0.0;

	this->kernel = NULL;
}

template<typename M>
MLPClassification<M>::MLPClassification(float l_rate, float r_rate, int n_iter, int n_classes, int n_dim, int n_hidden_units, Kernel<M>* kernel) {
	this->l_rate = l_rate;
	this->n_iter = n_iter;
	this->n_classes = n_classes;
	this->n_dim = n_dim;
	this->r_rate = r_rate;
	this->n_hidden_units = n_hidden_units;

	this->kernel = kernel;

	this->alpha = tensor<float, M>(extents[this->n_hidden_units][this->n_classes]);
	this->beta = tensor<float, M>(extents[1][this->n_classes]);

	this->w = tensor<float, M>(extents[this->n_dim][this->n_hidden_units]);
	this->b = tensor<float, M>(extents[1][this->n_hidden_units]);

	init();
}

template<typename M>
void MLPClassification<M>::init() {
	this->alpha = 0.f;
	this->beta = 0.f;

	this->w = 0.f;
	this->b = 0.f;

//	add_rnd_normal(alpha);
	//add_rnd_normal(b);
	add_rnd_normal(w);
//	add_rnd_normal(beta);

	alpha = .0f;
	w -= 0.5f;
	w *= .01f;
	beta = .0f;
	b = .0f;
}

template<typename M>
void MLPClassification<M>::calcLinearEq(tensor<float, M>& result, const tensor<float, M>& X) {
	gramMatrix = tensor<float, M>(extents[X.shape(0)][this->n_hidden_units]);
	kernel->calculate(gramMatrix, X, this->w, this->b);
	prod(result, gramMatrix, alpha, 'n', 'n', 1.f, 0.f);
	matrix_plus_row(result, this->beta);
}

template<typename M>
void MLPClassification<M>::calcSigma(tensor<float,M>& result, const tensor<float, M>& linearEq) {
	tensor<float, M> denomi(extents[linearEq.shape(0)]);
	result = linearEq.copy();
	denomi = 0.f;
	apply_scalar_functor(result, SF_EXP);
	reduce_to_col(denomi, linearEq, RF_ADDEXP);
	matrix_divide_col(result, denomi);
}

template<typename M>
void MLPClassification<M>::calcSigmaReduced(tensor<float, M>& result, const tensor<float, M>& linearEq) {
	tensor<float, M> tmpResulst = linearEq.copy();
	tensor<float, M> denomi(extents[linearEq.shape(0)]);
	result = 0.f;
	denomi = 0.f;

	reduce_to_col(denomi, linearEq, RF_LOGADDEXP);
	denomi = denomi * -1.f;
	matrix_plus_col(tmpResulst, denomi);
	apply_scalar_functor(result, tmpResulst, SF_EXP);
}

template<typename M>
double MLPClassification<M>::calcGradientDescent_2C(const tensor<float, M>& X, const tensor<float, M>& Y) {
	float n_data = X.shape(0);

	tensor<float, M> d_alpha(this->alpha.shape());
	tensor<float, M> d_beta(this->beta.shape());
	tensor<float, M> d_w(this->w.shape());
	tensor<float, M> d_b(this->b.shape());

	tensor<float, M> derivative_h(extents[n_data][this->n_hidden_units]);
	tensor<float, M> derivative_o(extents[n_data][this->n_classes]);

	tensor<float, M> d_w_tmp(extents[n_data][this->n_hidden_units]);
	tensor<float, M> linear_eq(extents[n_data][this->n_classes]);


	calcLinearEq(linear_eq, X);

	//Calculating sigmoid
	apply_scalar_functor(linear_eq, SF_SIGM);
	matrix_plus_col(linear_eq, -1.f*Y);

	apply_scalar_functor(derivative_o, linear_eq, SF_TANH);

	kernel->calculate_derivative(derivative_o, derivative_o);
	apply_binary_functor(linear_eq, linear_eq, derivative_o, BF_MULT);

	//calculating delta w
	kernel->calculate_derivative(derivative_h, gramMatrix);

	prod(d_w_tmp, linear_eq, this->alpha, 'n', 't');
	apply_binary_functor(d_w_tmp, d_w_tmp, derivative_h, BF_MULT);
	prod(d_w, X, d_w_tmp, 't', 'n', 1.f, 0.f);


	//calculating delta b
	reduce_to_row(d_b, d_w_tmp, RF_ADD, 2.f/n_data);


	//calculating delta alpha
	prod(d_alpha,  gramMatrix, linear_eq, 't', 'n', 2.f/n_data, 0.f);

	//calculating delta beta
	reduce_to_row(d_beta, linear_eq, RF_ADD, 2.f/n_data);


	//update alpha and beta
	alpha -= l_rate * d_alpha  + r_rate * this->alpha;
	beta -= l_rate * d_beta  + r_rate * this->beta;

	//update w and b
	w -= l_rate * d_w  + r_rate * this->w;
	b -= l_rate * d_b  + r_rate * this->b;

	return isConverging(0.008, d_alpha);
}

template<typename M>
double MLPClassification<M>::calcGradientDesc_MC(const tensor<float, M>& X, const tensor<float, M>& Y) {
	float n_data = X.shape(0);

	tensor<float, M> d_alpha(this->alpha.shape());
	tensor<float, M> d_beta(this->beta.shape());
	tensor<float, M> d_w(this->w.shape());
	tensor<float, M> d_b(this->b.shape());

	tensor<float, M> derivative_h(extents[n_data][this->n_hidden_units]);
	tensor<float, M> derivative_o(extents[n_data][this->n_classes]);

	tensor<float, M> d_w_tmp(extents[n_data][this->n_hidden_units]);
	tensor<float, M> linear_eq(extents[n_data][this->n_classes]);
	tensor<float, M> sigma(extents[n_data][this->n_classes]);

	calcLinearEq(linear_eq, X);

	//Calculating sigmoid
	calcSigmaReduced(sigma, linear_eq);

	//result = maximum(calcError(sigma, Y));
	sigma = sigma - Y;

	apply_scalar_functor(derivative_o, linear_eq, SF_TANH);

	kernel->calculate_derivative(derivative_o, derivative_o);
	apply_binary_functor(linear_eq, linear_eq, derivative_o, BF_MULT);

	//calculating delta w
	kernel->calculate_derivative(derivative_h, gramMatrix);

	prod(d_w_tmp, linear_eq, this->alpha, 'n', 't');
	apply_binary_functor(d_w_tmp, d_w_tmp, derivative_h, BF_MULT);
	prod(d_w, X, d_w_tmp, 't', 'n', 1.f, 0.f);


	//calculating delta b
	reduce_to_row(d_b, d_w_tmp, RF_ADD, 2.f/n_data);

	//calcualting delta alpha
	prod(d_alpha,  gramMatrix, linear_eq, 't', 'n', 2.f/n_data, 0.f);

	//calculatind delta betha
	reduce_to_row(d_beta, sigma, RF_ADD, 2.f/X.shape(0));

	//update alpha and beta
	alpha -= l_rate * d_alpha  + this->r_rate * this->alpha;
	beta -= l_rate * d_beta  + this->r_rate * this->beta;

	//update w and b
	w -= l_rate * d_w + r_rate * this->w;
	b -= l_rate * d_b + r_rate * this->b;

	return isConverging(0.001, d_alpha);;
}

template<typename M>
void MLPClassification<M>::fit(const tensor<float, M>& X, const tensor<float, M>& Y ) {
	tensor<float, M> Y_Multi = convertYtoM(Y);
	//this->X = X;
//	this->alpha = tensor<float, M>(extents[X.shape(0)][n_classes]);
//	this->alpha = 0.f;

	int iter = 0;
	double con = 0.0;
//	int miss = 0;

	do {
		if (this->n_classes == 2)
			con = calcGradientDescent_2C(X, Y);
		else
			con = calcGradientDesc_MC(X, Y_Multi);
//		miss = missClassified(X, Y);
		cout<<"iter: "<<iter<<" "<<con<<" MissClass Train# "<<missClassified(X, Y)<<endl;
		iter++;
	} while ((iter < this->n_iter));
//} while ((iter < this->n_iter) && (con > 0.00001));
}

template<typename M>
tensor<float, M> MLPClassification<M>::predict(const tensor<float, M>& X_test) {
	tensor<float, M> result;
	if (this->n_classes == 2)
		result = predict_2C(X_test);
	else
		result = predict_MC(X_test);
	return result;
}

template<typename M>
void MLPClassification<M>::predict(const tensor<float, M>& X_test, char* fileName) {
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
tensor<float, M> MLPClassification<M>::predict_2C(const tensor<float, M>& X_test) {
	tensor<float, M> result(X_test.shape());
	tensor<float, M> linear_eq(extents[X_test.shape(0)][this->n_classes]);

	calcLinearEq(linear_eq, X_test);

	apply_scalar_functor(linear_eq, linear_eq, SF_SIGM);
	apply_scalar_functor(result, linear_eq, SF_GT, .5f);

	return result[indices[index_range()][index_range(0, 1)]].copy();
}

template<typename M>
tensor<float, M> MLPClassification<M>::predict_MC(const tensor<float, M>& X_test) {
	tensor<float, M> result(extents[X_test.shape(0)][1]);
	tensor<float, M> linear_eq(extents[X_test.shape(0)][this->n_classes]);
	tensor<float, M> sigma(extents[X_test.shape(0)][this->n_classes]);

	calcLinearEq(linear_eq, X_test);

	calcSigmaReduced(sigma, linear_eq);

	reduce_to_col(result, sigma, RF_ARGMAX);

	return result[indices[index_range()][index_range(0, 1)]].copy();
}

template<typename M>
tensor<float, M> MLPClassification<M>::convertYtoM(const tensor<float, M>& Y) {
	tensor<float, host_memory_space> result_h(extents[Y.shape(0)][n_classes]);
	tensor<float, M> result(extents[Y.shape(0)][n_classes]);
	result_h = 0.f;

	tensor<float, host_memory_space> y_host(Y.shape());
	y_host = Y;

	for (unsigned i = 0, ii = Y.shape(0); i < ii; ++i) {
		// use the class number as index to set the cell to 1
		result_h(i, y_host[i]) = 1.f;
	}
	result = result_h;

	return result;
}

template<typename M>
double MLPClassification<M>::isConverging(const float& epsilon, const tensor<float, M>& delta_w) {
	double result = 0.0;
	tensor_view<float, M> slice = delta_w[indices[index_range()][index_range(0, 1)]];
	tensor<float, M> n = slice.copy();
	 result = norm2(slice);
//	cout<<"delta_W\n"<<delta_w<<endl;
//	cout<<"Slice\n"<<n<<endl;
	return result;
}

template<typename M>
tensor<float, M> MLPClassification<M>::calcError(const tensor<float, M>& sigma, const tensor<float, M>& Y) {
	tensor<float, M> sigma_1 = sigma.copy();
	tensor<float, M> sigma_0 = 1.f - sigma;
	tensor<float, M> tmpSigma(sigma.shape());
	tensor<float, M> result(extents[n_classes]);

	for (int i = 0; i < this->n_classes; i++) {
		tensor_view<float, M> slice_Y1 = Y[indices[index_range()][index_range(i, i+1)]];
		tensor_view<float, M> slice_S1 = sigma_1[indices[index_range()][index_range(i, i+1)]];
		tensor_view<float, M> slice_S0 = sigma_0[indices[index_range()][index_range(i, i+1)]];

		tensor<float, M> yy = slice_Y1.copy();
		tensor<float, M> ss1 = slice_S1.copy();
		tensor<float, M> ss0 = slice_S0.copy();
		apply_scalar_functor(ss1, SF_LOG);
		apply_scalar_functor(ss0, SF_LOG);
		matrix_times_col(ss1, yy[indices[index_range()][0]]);
		matrix_times_col(ss0, 1.f - yy[indices[index_range()][0]]);
		sigma_1[indices[index_range()][index_range(i, i+1)]] = ss1;
		sigma_0[indices[index_range()][index_range(i, i+1)]] = ss0;
	}

	tmpSigma = sigma_1 + sigma_0;
	reduce_to_row(result, tmpSigma);
	return -1.f * result;
}

template<typename M>
int MLPClassification<M>::missClassified(const tensor<float, M>& X, const tensor<float, M>& Y) {
	int result = 0;
	tensor<float, M> predicted = predict(X);
	tensor<float, M> diff(Y.shape());
	apply_binary_functor(diff, Y, predicted, BF_EQ);
	result = Y.shape(0) - sum(diff);
	return result;
}

template<typename M>
tensor<float, M> MLPClassification<M>::getAlpha() {
	return this->alpha;
}

template<typename M>
tensor<float, M> MLPClassification<M>::getBeta() {
	return this->beta;
}

template<typename M>
tensor<float, M> MLPClassification<M>::getW() {
	return this->w;
}

template<typename M>
tensor<float, M> MLPClassification<M>::getB() {
	return this->b;
}

template<typename M>
double MLPClassification<M>::predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test) {
	double result = 0.0;
	result = missClassified(X_test, Y_test);
	return result;
}

template<typename M>
void MLPClassification<M>::printParamToScreen() {

}

template<typename M>
MLPClassification<M>::~MLPClassification() {

}

template class MLPClassification<dev_memory_space>;
template class MLPClassification<host_memory_space>;
