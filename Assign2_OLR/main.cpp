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
#define N_DIM 2
#define N_TRAINING 200
#define N_ITERATION 200
#define N_TEST 50
#define MEMORY dev_memory_space

int main(){

	int i = 1;
	do {
		tensor<float, MEMORY> w_0(extents[N_DIM][i]);
		tensor<float, MEMORY> b_0(extents[1][i]);
		initialize_mersenne_twister_seeds(time(NULL));
		add_rnd_normal(w_0);
		add_rnd_normal(b_0);

		DataSet<MEMORY> dsTrain(N_DIM, N_TRAINING, i, w_0, b_0);
		dsTrain.createData();

		DataSet<MEMORY> dsTest(N_DIM, N_TEST, i, w_0, b_0);
		dsTest.createData();

		double time = 0.0;
		for (int j = 0; j<=50; j++) {
			{
			OnLineLinearRegression<MEMORY> o(0.0009, N_ITERATION, i, N_DIM);
			clock_t start = clock();
			o.fit(dsTrain.getData(), dsTrain.getLabels());
			o.predict(dsTest.getData());
			time += (double)(clock()-start)/CLOCKS_PER_SEC;
			}
		}
		cout<<i<<" "<<time/50<<endl;
		i+=500;
	} while (i<=10000);

//	cout<<"RESULT\n"<<result<<endl;


	return 0;
}
