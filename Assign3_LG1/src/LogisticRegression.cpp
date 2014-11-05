/*
 * LogisticRegression.cpp
 *
 *  Created on: May 16, 2014
 *      Author: Kostadin Cvejoski
 */

#include "LogisticRegression.h"

template<typename M>
LogisticRegression<M>::LogisticRegression() {
	this->n_iter = 0;
	this->n_dim = 0;
	this->n_classes = 0;
	this->l_rate = 0.0;
	this->r_rate = 0.0;

	this->b = NULL;
	this->w = NULL;
}

template<typename M>
LogisticRegression<M>::LogisticRegression(float l_rate, float r_rate, int n_iter, int n_classes, int n_dim) {
	this->l_rate = l_rate;
	this->n_iter = n_iter;
	this->n_classes = n_classes;
	this->n_dim = n_dim;
	this->r_rate = r_rate;

	this->w = tensor<float, M>(extents[n_dim][n_classes]);
	this->b = tensor<float, M>(extents[n_classes]);

	init();
}

template<typename M>
tensor<float, M> LogisticRegression<M>::calcLinearEq(const tensor<float, M>& X) {
	tensor<float, M> result;
	result = tensor<float, M>(extents[X.shape(0)][this->n_classes]);
	prod(result, X, w, 'n', 'n', 1.f, 0.f);
	matrix_plus_row(result, b);

	return result;
}

template<typename M>
tensor<float, M> LogisticRegression<M>::calcSigma(const tensor<float, M>& linearEq) {
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
tensor<float, M> LogisticRegression<M>::calcSigmaReduced(const tensor<float, M>& linearEq) {
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
double LogisticRegression<M>::calcGradientDescent_2C(const tensor<float, M>& X, const tensor<float, M>& Y) {
	tensor<float, M> d_w(this->w.shape());
	tensor<float, M> d_b(this->b.shape());

	float n_data = X.shape(0);

	tensor<float, M> linear_eq = calcLinearEq(X);

	//Calculating sigmoid
	apply_scalar_functor(linear_eq, SF_SIGM);
	matrix_plus_col(linear_eq, -1.f*Y);

	//calcualting delta w
	prod(d_w, X, linear_eq, 't', 'n', 1.f, 0.f);

	//calculatind delta b
	reduce_to_row(d_b, linear_eq);


	//update w and b
	w -= l_rate * d_w / n_data + r_rate * w;
	b -= l_rate * d_b / n_data + r_rate * b;

	return isConverging(0.008, d_w);
}

template<typename M>
double LogisticRegression<M>::calcGradientDesc_MC(const tensor<float, M>& X, const tensor<float, M>& Y) {
	tensor<float, M> d_w(this->w.shape());
	tensor<float, M> d_b(this->b.shape());

	double result = 0.0;

	float n_data = X.shape(0);

	tensor<float, M> linear_eq = calcLinearEq(X);

	//Calculating sigmoid
	tensor<float, M> sigma = calcSigmaReduced(linear_eq);
	//result = maximum(calcError(sigma, Y));
	sigma = sigma - Y;

	//calcualting delta w
	prod(d_w, X, sigma, 't', 'n', 1.f, 0.f);

	//calculatind delta b
	reduce_to_row(d_b, sigma);

	d_w /= n_data;
	d_b /= n_data;

	//update w and b
	w -= l_rate * d_w + r_rate * w;
	b -= l_rate * d_b + r_rate * b;

	result = isConverging(0.001, d_w);

	return result;
}

template<typename M>
void LogisticRegression<M>::init() {
	initialize_mersenne_twister_seeds(time(NULL));
	this->w = 0.f;
	this->b = 0.f;
	add_rnd_normal(w);
	add_rnd_normal(b);

	w *= .01f;
	b *= .01f;
}

template<typename M>
void LogisticRegression<M>::fit(const tensor<float, M>& X, const tensor<float, M>& Y) {
	tensor<float, M> Y_Multi = convertYtoM(Y);

	int iter = 0;
	double con = 0.0;

	do {
		if (this->n_classes == 2)
			con = calcGradientDescent_2C(X, Y);
		else
			con = calcGradientDesc_MC(X, Y_Multi);
			//cout<<"iter: "<<iter<<" "<<con<<" MissClass # "<<endl;
			cout<<"iter: "<<iter<<" "<<con<<" MissClass # "<<missClassified(X, Y)<<endl;
		iter++;
	} while ((iter < this->n_iter) );
}

template<typename M>
tensor<float, M> LogisticRegression<M>::predict(const tensor<float, M>& X_test) {
	tensor<float, M> result;
	if (this->n_classes == 2)
		result = predict_2C(X_test);
	else
		result = predict_MC(X_test);
	return result;
}

template<typename M>
tensor<float, M> LogisticRegression<M>::predict_2C(const tensor<float, M>& X_test) {
	tensor<float, M> result(X_test.shape());
	tensor<float, M> linear_eq = calcLinearEq(X_test);

	apply_scalar_functor(linear_eq, linear_eq, SF_SIGM);
	apply_scalar_functor(result, linear_eq, SF_GT, .5f);

	return result[indices[index_range()][index_range(0, 1)]].copy();
}

template<typename M>
tensor<float, M> LogisticRegression<M>::predict_MC(const tensor<float, M>& X_test) {
	tensor<float, M> result(extents[X_test.shape(0)][1]);
	tensor<float, M> linear_eq = calcLinearEq(X_test);

	tensor<float, M> sigma = calcSigmaReduced(linear_eq);
	reduce_to_col(result, sigma, RF_ARGMAX);

	return result[indices[index_range()][index_range(0, 1)]].copy();
}

template<typename M>
tensor<float, M> LogisticRegression<M>::convertYtoM(const tensor<float, M>& Y) {
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
double LogisticRegression<M>::isConverging(const float& epsilon, const tensor<float, M>& delta_w) {
	double result = 0.0;
	tensor_view<float, M> slice = delta_w[indices[index_range()][index_range(0, 1)]];
	tensor<float, M> n = slice.copy();
	 result = norm2(slice);
//	cout<<"delta_W\n"<<delta_w<<endl;
//	cout<<"Slice\n"<<n<<endl;
	return result;
}

template<typename M>
tensor<float, M> LogisticRegression<M>::calcError(const tensor<float, M>& sigma, const tensor<float, M>& Y) {
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
int LogisticRegression<M>::missClassified(const tensor<float, M>& X, const tensor<float, M>& Y) {
	int result = 0;
	tensor<float, M> predicted = predict(X);
	tensor<float, M> diff(Y.shape());
	apply_binary_functor(diff, Y, predicted, BF_EQ);
	result = Y.shape(0) - sum(diff);
	return result;
}


template<typename M>
tensor<float, M> LogisticRegression<M>::getW() {
	return w;
}

template<typename M>
tensor<float, M> LogisticRegression<M>::getB() {
	return b;
}

template<typename M>
void LogisticRegression<M>::printParamToScreen() {
	cout<<"W: \n"<<getW()<<endl;
	cout<<"B: \n"<<getB()<<endl;
	//plotLearnedWeights();
}

template<typename M>
double LogisticRegression<M>::predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test) {
	double result = 0.0;
	result = missClassified(X_test, Y_test);
	return result;
}

template<typename M>
LogisticRegression<M>::~LogisticRegression() {
	// TODO Auto-generated destructor stub
}

template class LogisticRegression<host_memory_space>;
template class LogisticRegression<dev_memory_space>;
//template class MLalgorithm<dev_memory_space>;
//template class MLalgorithm<host_memory_space>;
