/*
 * main.cpp
 *
 *  Created on: May 1, 2014
 *      Author: cve
 */

#include <time.h>
#include "src/OnLineLinearRegression.h"
#include "src/DataSet.h"

#define N_OUTPUTS 1
#define N_DIM 1
#define N_TRAINING 50
#define N_ITERATION 4000
#define N_TEST 30
#define MEMORY dev_memory_space

int main(){

		tensor<float, MEMORY> w_0(extents[N_DIM][N_OUTPUTS]);
		tensor<float, MEMORY> b_0(extents[1][N_OUTPUTS]);
		initialize_mersenne_twister_seeds(0);

		fill_rnd_uniform(w_0);
		fill_rnd_uniform(b_0);

		DataSet<MEMORY> dsTrain(N_DIM, N_TRAINING, N_OUTPUTS, w_0, b_0);
		dsTrain.createData();

		DataSet<MEMORY> dsTest(N_DIM, N_TEST, N_OUTPUTS, w_0, b_0);
		dsTest.createData();

		OnLineLinearRegression<MEMORY> o(0.0005, N_ITERATION, N_OUTPUTS, N_DIM);

		o.fit(dsTrain.getData(), dsTrain.getLabels());
		o.predict(dsTest.getData());



		cout<<w_0<<endl;
		cout<<b_0<<endl;
		cout<<o.getW()<<endl;
		cout<<o.getB()<<endl;



	return 0;
}
