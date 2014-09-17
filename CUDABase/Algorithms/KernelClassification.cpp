/*
 * KernelClassification.cpp
 *
 *  Created on: Jul 8, 2014
 *      Author: cve
 */

#include "KernelClassification.h"

template <typename M>
KernelClassification<M>::KernelClassification() {
	this->n_iter = 0;
	this->n_dim = 0;
	this->n_classes = 0;
	this->l_rate = 0.0;
	this->r_rate = 0.0;
	this->kernel = NULL;
}

template <typename M>
KernelClassification<M>::KernelClassification(float l_rate, float r_rate, int n_iter, int n_classes, int n_dim, Kernel<M>* kernel) {
	this->l_rate = l_rate;
	this->n_iter = n_iter;
	this->n_classes = n_classes;
	this->n_dim = n_dim;
	this->r_rate = r_rate;

	this->kernel = kernel;

	this->b = tensor<float, M>(extents[1][n_classes]);
	this->b = 0.f;
}

template<typename M>
tensor<float, M> KernelClassification<M>::calcLinearEq(const tensor<float, M>& X_test) {
	tensor<float, M> result(extents[X_test.shape(0)][this->n_classes]);
	tensor<float, M> result_t(extents[this->n_classes][X_test.shape(0)]);
	gramMatrix = tensor<float, M>(extents[X.shape(0)][X_test.shape(0)]);

	kernel->calculate(gramMatrix, X, X_test);
	prod(result_t, alpha, gramMatrix, 't', 'n', 1.f, 0.f);
	transpose(result, result_t);
	matrix_plus_row(result, b);

	return result;
}

template<typename M>
tensor<float, M> KernelClassification<M>::calcSigma(const tensor<float, M>& linearEq) {
	tensor<float, M> result(linearEq.shape());
	tensor<float, M> denomi(extents[linearEq.shape(0)]);
	tensor<float, M> tmpResulst = linearEq.copy();
	result = 0.f;
	denomi = 0.f;

	apply_scalar_functor(tmpResulst, SF_EXP);
	reduce_to_col(denomi, linearEq, RF_ADDEXP);
	matrix_divide_col(tmpResulst, denomi);

	return tmpResulst;
}

template<typename M>
tensor<float, M> KernelClassification<M>::calcSigmaReduced(const tensor<float, M>& linearEq) {
	tensor<float, M> result(linearEq.shape());
	tensor<float, M> tmpResulst = linearEq.copy();
	tensor<float, M> denomi(extents[linearEq.shape(0)]);
	result = 0.f;
	denomi = 0.f;

	reduce_to_col(denomi, linearEq, RF_LOGADDEXP);
	denomi = denomi * -1.f;
	matrix_plus_col(tmpResulst, denomi);
	apply_scalar_functor(result, tmpResulst, SF_EXP);

	return result;
}

template<typename M>
double KernelClassification<M>::calcGradientDescent_2C(const tensor<float, M>& X, const tensor<float, M>& Y) {
	tensor<float, M> d_alpha(this->alpha.shape());
	tensor<float, M> d_b(this->b.shape());

	float n_data = X.shape(0);

	tensor<float, M> linear_eq = calcLinearEq(X);

	//Calculating sigmoid
	apply_scalar_functor(linear_eq, SF_SIGM);
	matrix_plus_col(linear_eq, -1.f*Y);

	//calcualting delta a
	prod(d_alpha, gramMatrix, linear_eq, 'n', 'n', 2.f/n_data, 0.f);

	//calculatind delta b
	reduce_to_row(d_b, linear_eq, RF_ADD, 2.f/n_data);


	//update w and b
	alpha -= l_rate * d_alpha  + r_rate * this->alpha;
	b -= l_rate * d_b + r_rate * b;

	return isConverging(0.008, d_alpha);
}

template<typename M>
double KernelClassification<M>::calcGradientDesc_MC(const tensor<float, M>& X, const tensor<float, M>& Y) {
	tensor<float, M> d_alpha(this->alpha.shape());
	tensor<float, M> d_b(this->b.shape());

	double result = 0.0;

	float n_data = X.shape(0);

	tensor<float, M> linear_eq = calcLinearEq(X);

	//Calculating sigmoid
	tensor<float, M> sigma = calcSigmaReduced(linear_eq);
	//result = maximum(calcError(sigma, Y));
	sigma = sigma - Y;

	//calcualting delta w
	prod(d_alpha, gramMatrix, sigma, 'n', 'n', 2.f/n_data, 0.f);

	//calculatind delta b
	reduce_to_row(d_b, sigma, RF_ADD, 2.f/n_data);

	//update w and b
	alpha -= l_rate * d_alpha + r_rate * alpha / n_data;
	b -= l_rate * d_b + r_rate * b / n_data;

	result = isConverging(0.001, d_alpha);

	return result;
}

template<typename M>
void KernelClassification<M>::init() {
		this->alpha = 0.f;
	this->b = 0.f;
	add_rnd_normal(alpha);
	add_rnd_normal(b);
	alpha *= .01f;
	b *= .01f;
}

template<typename M>
void KernelClassification<M>::fit(const tensor<float, M>& X, const tensor<float, M>& Y ) {
	tensor<float, M> Y_Multi = convertYtoM(Y);
	this->X = X;
	this->alpha = tensor<float, M>(extents[X.shape(0)][n_classes]);
	this->alpha = 0.f;

	int iter = 0;
	double con = 0.0;
//	int miss = 0;

	do {
		if (this->n_classes == 2)
			calcGradientDescent_2C(X, Y);
		else
			 calcGradientDesc_MC(X, Y_Multi);
//		miss = missClassified(X, Y);
		cout<<"iter: "<<iter<<" "<<con<<" MissClass Train# "<<missClassified(X, Y)<<endl;
		iter++;
	} while ((iter < this->n_iter) );
}

template<typename M>
tensor<float, M> KernelClassification<M>::predict(const tensor<float, M>& X_test) {
	tensor<float, M> result;
	if (this->n_classes == 2)
		result = predict_2C(X_test);
	else
		result = predict_MC(X_test);
	return result;
}

template<typename M>
void KernelClassification<M>::predict(const tensor<float, M>& X_test, char* fileName) {
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
tensor<float, M> KernelClassification<M>::predict_2C(const tensor<float, M>& X_test) {
	tensor<float, M> result(X_test.shape());
	tensor<float, M> linear_eq = calcLinearEq(X_test);

	apply_scalar_functor(linear_eq, linear_eq, SF_SIGM);
	apply_scalar_functor(result, linear_eq, SF_GT, .5f);

	return result[indices[index_range()][index_range(0, 1)]].copy();
}

template<typename M>
tensor<float, M> KernelClassification<M>::predict_MC(const tensor<float, M>& X_test) {
	tensor<float, M> result(extents[X_test.shape(0)][1]);
	tensor<float, M> linear_eq = calcLinearEq(X_test);

	tensor<float, M> sigma = calcSigmaReduced(linear_eq);
	reduce_to_col(result, sigma, RF_ARGMAX);

	return result[indices[index_range()][index_range(0, 1)]].copy();
}

template<typename M>
tensor<float, M> KernelClassification<M>::convertYtoM(const tensor<float, M>& Y) {
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
double KernelClassification<M>::isConverging(const float& epsilon, const tensor<float, M>& delta_w) {
	double result = 0.0;
	tensor_view<float, M> slice = delta_w[indices[index_range()][index_range(0, 1)]];
	tensor<float, M> n = slice.copy();
	 result = norm2(slice);
//	cout<<"delta_W\n"<<delta_w<<endl;
//	cout<<"Slice\n"<<n<<endl;
	return result;
}

template<typename M>
tensor<float, M> KernelClassification<M>::calcError(const tensor<float, M>& sigma, const tensor<float, M>& Y) {
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
int KernelClassification<M>::missClassified(const tensor<float, M>& X, const tensor<float, M>& Y) {
	int result = 0;
	tensor<float, M> predicted = predict(X);
	tensor<float, M> diff(Y.shape());
	apply_binary_functor(diff, Y, predicted, BF_EQ);
	result = Y.shape(0) - sum(diff);
	return result;
}


template<typename M>
tensor<float, M> KernelClassification<M>::getAlpha() {
	return this->alpha;
}

template<typename M>
tensor<float, M> KernelClassification<M>::getB() {
	return b;
}



template<typename M>
double KernelClassification<M>::predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test) {
	double result = 0.0;
	result = missClassified(X_test, Y_test);
	return result;
}

template<typename M>
void KernelClassification<M>::printParamToScreen() {

}



template <typename M>
KernelClassification<M>::~KernelClassification() {
	// TODO Auto-generated destructor stub
}

//template class KernelClassification<dev_memory_space>;
//template class KernelClassification<host_memory_space>;
