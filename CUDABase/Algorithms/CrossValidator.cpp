/*
 * CrossValidator.cpp
 *
 *  Created on: May 20, 2014
 *      Author: cve
 */

#include "CrossValidator.h"

template<typename M>
CrossValidator<M>::CrossValidator() {
	this->k_fold = 0;
	this->nClasses = 0;
	this->nIterations = 0;
	this->bestModel = NULL;
	this->factory = NULL;
}
template<typename M>
CrossValidator<M>::CrossValidator(const vector<float>& l_rates, const vector<float>& r_rates, const unsigned& nClasses, const unsigned& nIterations, const unsigned& k_fold, Factory<M> * factory) {
	this->k_fold = k_fold;
	this->l_rates = l_rates;
	this->r_rates = r_rates;
	this->nClasses = nClasses;
	this->factory = factory;
	this->nIterations = nIterations;
	this->bestModel = NULL;
}

template<typename M>
CrossValidator<M>::CrossValidator(const vector<float>& l_rates, const vector<float>& r_rates, const vector<float>& sigma, const unsigned& nClasses, const unsigned& nIterations, const unsigned& k_fold, Factory<M> * factory) {
	this->k_fold = k_fold;
	this->l_rates = l_rates;
	this->r_rates = r_rates;
	this->sigma = sigma;
	this->nClasses = nClasses;
	this->factory = factory;
	this->nIterations = nIterations;
	this->bestModel = NULL;
}


template<typename M>
float CrossValidator<M>::fitModel(const tensor<float, M>& X, const tensor<float, M>& Y, MLalgorithm<M>* model) {
	vector<tensor<float, M> > splitX;
	vector<tensor<float, M> > splitY;
	float error = 0.0;
	for (unsigned i = 0; i < this->k_fold; i++) {
		splitX = kSplitAlternative(X, i);
		splitY = kSplitAlternative(Y, i);
		model->fit(splitX.at(1), splitY.at(1));
		error += model->predictWithError(splitX.at(0), splitY.at(0));
		splitX.clear();
		splitY.clear();
	}
	return error / k_fold;
}

template<typename M>
void CrossValidator<M>::fitHyperParamethers(const tensor<float, M>& X,
		vector<float>& meanError, const tensor<float, M>& Y,
		vector<MLalgorithm<M> *>& models) {
	for (unsigned i = 0; i < this->l_rates.size(); i++) {
		for (unsigned j = 0; j < this->r_rates.size(); j++) {
			float l_rate = this->l_rates.at(i);
			float r_rate = this->r_rates.at(j);
			MLalgorithm<M>* model = factory->generate(l_rate, r_rate,
					nIterations, nClasses, X.shape(1));
			meanError.push_back(fitModel(X, Y, model));
			models.push_back(model);
		}
	}
}

template<typename M>
void CrossValidator<M>::fitHyperParamethersSigma(const tensor<float, M>& X,
		vector<float>& meanError, const tensor<float, M>& Y,
		vector<MLalgorithm<M> *>& models) {
	for (unsigned i = 0; i < this->l_rates.size(); i++) {
		for (unsigned j = 0; j < this->r_rates.size(); j++) {
			for (unsigned q = 0; q < this->sigma.size(); q++) {
				float l_rate = this->l_rates.at(i);
				float r_rate = this->r_rates.at(j);
				float sigma = this->sigma.at(q);
				cout<<"MODEL :"<<i+j+q<<endl;
				cout<<" eta: "<<(l_rates.at(i));
				cout<<" lambda: "<<r_rates.at(j);
				cout<<" sigma: "<<sigma<<" Error: ";
				MLalgorithm<M>* model = factory->generate(l_rate, r_rate, sigma,
					nIterations, nClasses, X.shape(1));
				meanError.push_back(fitModel(X, Y, model));
				cout<<meanError.at(meanError.size()-1)<<endl;
				models.push_back(model);
			}
		}
	}
}

template<typename M>
MLalgorithm<M>* CrossValidator<M>::findBestModel(const vector<float>& meanError, const vector<MLalgorithm<M>* >& models) {
	float minError = INT_MAX;
	unsigned bestModel = -1;
	for (unsigned i = 0; i < meanError.size(); i++) {
		if (minError > meanError.at(i)) {
			minError = meanError.at(i);
			bestModel = i;
		}
	}
	return models.at(bestModel);
}

template<typename M>
void CrossValidator<M>::fit(const tensor<float, M>& X, const tensor<float, M>& Y) {
	vector<MLalgorithm<M>* > models;
	vector<float> meanError;
	fitHyperParamethers(X, meanError, Y, models);
	this->bestModel = findBestModel(meanError, models);
	this->bestModel->fit(X, Y);
	for (unsigned i = 0; i < l_rates.size(); i++) {
		for (unsigned j = 0; j < r_rates.size(); j++)
			cout<<"MODEL :"<<i+j<<" eta: "<<(l_rates.at(i))<<" lambda: "<<r_rates.at(j)<<" error # "<<meanError.at(i+j)<<endl;
	}
}

template<typename M>
void CrossValidator<M>::fit_kernel(const tensor<float, M>& X, const tensor<float, M>& Y) {
	vector<MLalgorithm<M>* > models;
	vector<float> meanError;
	fitHyperParamethersSigma(X, meanError, Y, models);

	this->bestModel = findBestModel(meanError, models);
	this->bestModel->fit(X, Y);
	cout<<"BEST MODEL PARAMETERS:\n";
	bestModel->printParamToScreen();

//	for (unsigned i = 0; i < l_rates.size(); i++) {
//		for (unsigned j = 0; j < r_rates.size(); j++)
//			cout<<"MODEL :"<<i+j<<" eta: "<<(l_rates.at(i))<<" lambda: "<<r_rates.at(j)<<" error # "<<meanError.at(i+j)<<endl;
//	}
//

}

template<typename M>
vector<tensor_view<float, M> > CrossValidator<M>::kSplit(const tensor<float, M>& X) {
	vector<tensor_view<float, M> > result;
	unsigned portion = X.shape(0) / this->k_fold;
	for (unsigned i = 0; i < this->k_fold; i++) {
		tensor_view<float, M> tv = X[indices[index_range(i * portion, (i + 1) * portion )]];
		result.push_back(tv);
	}
	return result;
}

template<typename M>
vector<tensor<float, M> > CrossValidator<M>::kSplitAlternative(const tensor<float, M>& X, const unsigned int& i) {
	vector<tensor<float, M> > result;
	unsigned portion = X.shape(0) / this->k_fold;
	tensor<float, M> test;
	tensor<float, M> train(extents[X.shape(0) - portion][X.shape(1)]);

	if (i == 0) {
		tensor_view<float, M> tv_test = X[indices[index_range(i * portion, (i + 1) * portion )]];
		tensor_view<float, M> tv_train = X[indices[index_range((i + 1) * portion, X.shape(0) )]];
		train = tv_train.copy();
		test = tv_test.copy();
	} else if ( i == k_fold - 1) {
		tensor_view<float, M> tv_test = X[indices[index_range( i * portion, X.shape(0) )]];
		tensor_view<float, M> tv_train = X[indices[index_range( 0, i * portion )]];
		train = tv_train.copy();
		test = tv_test.copy();
	} else {
		tensor_view<float, M> tv_test = X[indices[index_range(i * portion, (i + 1) * portion )]];
		tensor_view<float, M> tv_train = X[indices[index_range( 0, i * portion )]];
		train[indices[index_range( 0, i * portion )]] = tv_train.copy();
		tensor_view<float, M> tv_train1 = X[indices[index_range( (i + 1) * portion, X.shape(0) )]];
		train[indices[index_range( i * portion, train.shape(0) )]] =  tv_train1.copy();
		test = tv_test.copy();
	}

	result.push_back(test);
	result.push_back(train);
	return result;
}

template<typename M>
tensor<float, M> CrossValidator<M>::predict(const tensor<float, M>& X_test) {
	tensor<float, M> result;
	result = this->bestModel->predict(X_test);
	return result;
}

template<typename M>
double CrossValidator<M>::predictWithError(const tensor<float, M>& X_test, const tensor<float, M>& Y_test) {
	double result = 0.0;
	result = bestModel->predictWithError(X_test, Y_test);
	return result;
}

template<typename M>
void CrossValidator<M>::printBestModel() {
	bestModel->printParamToScreen();
}

template<typename M>
CrossValidator<M>::~CrossValidator() {

}

template class CrossValidator<dev_memory_space>;
template class CrossValidator<host_memory_space>;
