/*
 * main.cpp
 *
 *  Created on: May 1, 2014
 *      Author: cve
 */


#include "src/OnLineLinearRegression.h"
#include "src/DataSet.h"

#define N_OUTPUTS 2
#define N_DIM 2
#define N_TRAINING 200
#define N_ITERATION 200
#define N_TEST 50
#define MEMORY host_memory_space

int main(){

	tensor<float, host_memory_space> w_0(extents[N_DIM][N_OUTPUTS]);
	tensor<float, host_memory_space> b(extents[1][N_OUTPUTS]);

	add_rnd_normal(w_0);
	add_rnd_normal(b);

	DataSet<host_memory_space> ds(N_DIM, N_TRAINING, N_OUTPUTS, w_0, b);
	ds.createData();

	DataSet<host_memory_space> dsTest(N_DIM, N_TEST, N_OUTPUTS, w_0, b);
	dsTest.createData();



	OnLineLinearRegression<host_memory_space> o(0.0009, N_ITERATION, N_OUTPUTS, N_DIM);
	o.fit(ds.getData(), ds.getLabels());

	tensor<float, host_memory_space> pred = o.predict(dsTest.getData());
	tensor<float, host_memory_space> result(dsTest.getLabels().shape());

	apply_binary_functor(result, pred, dsTest.getLabels(), BF_SUBTRACT);

//	cout<<"RESULT\n"<<result<<endl;


	return 0;
}
